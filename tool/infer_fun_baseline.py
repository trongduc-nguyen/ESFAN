
# infer_fun_baseline.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
from tqdm import tqdm
import cv2
from skimage import morphology

from .baseline_dataset import BaselineInferDataset
from . import infer_utils

# --- HÀM PHỤ TRỢ ĐỂ TÌM ĐƯỜNG DẪN ẢNH ĐÚNG ---
def get_image_path(dataroot, img_name):
    # Kiểm tra xem ảnh nằm trong thư mục 'img/' hay nằm ngay ở root
    path_in_subdir = os.path.join(dataroot, 'img', img_name + '.png')
    path_in_root = os.path.join(dataroot, img_name + '.png')
    
    if os.path.exists(path_in_subdir):
        return path_in_subdir
    elif os.path.exists(path_in_root):
        return path_in_root
    else:
        return path_in_subdir

def infer_baseline(model, dataroot, n_class, args, generate_imges=False):
    method = getattr(args, 'method', 'baseline')
    print(f"Executing Inference with Method: [{method.upper()}]")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = BaselineInferDataset(dataroot=dataroot, n_class=n_class, transform=transform)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, batch_size=1)
    
    pred_list = []
    pred_img = []
    gt_list = []

    for img_name, img_tensor, label_tensor in tqdm(infer_data_loader, desc=f"Inferring"):
        img_name = img_name[0]
        pred_img.append(img_name)
        image_label = label_tensor.to(device)
        img_tensor = img_tensor.to(device)
        
        orig_img_path = get_image_path(dataroot, img_name)
        
        try:
            # Load ảnh gốc để tính Heuristic
            orig_img_pil = Image.open(orig_img_path).convert("RGB")
            orig_img_np = np.array(orig_img_pil)
            orig_img_shape = orig_img_np.shape[:2] #(H, W)
        except Exception as e:
            print(f"Error reading image {orig_img_path}: {e}")
            continue

        gt_map_path_subdir = os.path.join(dataroot, 'mask/', img_name + '.png')
        if os.path.exists(gt_map_path_subdir):
            gt_map = np.array(Image.open(gt_map_path_subdir))
        else:
            gt_map = np.zeros(orig_img_shape, dtype=np.uint8)
            
        gt_list.append(gt_map)

        # Model Inference
        with torch.no_grad():
            output_dict = model(img_tensor)

        # Xử lý CAM
        if method == 'baseline':
            cam_final = output_dict['cam8']
            cam_upsampled = F.interpolate(cam_final, size=orig_img_shape, mode='bilinear', align_corners=False)[0]
        elif method == 'tgf_pr':
            cam_fused = infer_utils.tri_scale_gated_fusion(output_dict, target_size=orig_img_shape)
            cam_refined = infer_utils.prototype_refinement(
                cam_fused, output_dict['features'], image_label, target_size=orig_img_shape, n_class=n_class
            )
            cam_upsampled = cam_refined[0]
            # cam_upsampled = cam_fused[0]
        # Tính toán điểm số CAM cho các lớp mô (Foreground)
        cam_scores = cam_upsampled.cpu().numpy() * image_label.cpu().clone().view(n_class, 1, 1).numpy()
        
        # Argmax để lấy nhãn mô dự đoán ban đầu (0, 1, 2, 3)
        pred_mask = np.argmax(cam_scores, axis=0).astype(np.uint8)
        

        pred_mask[gt_map == 4] = 4
        
        pred_list.append(pred_mask)
    
    if generate_imges:
        return gt_list, pred_list, pred_img
    return gt_list, pred_list

def infer_sgps(alpha,beta,model_cam, model_patch, dataroot, n_class, args):
    """
    Thực hiện inference với SGPS (Kết hợp CAM + Patch Model).
    """
    print(f"Executing Inference with Method: [SGPS Refinement]")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Đăng ký Hook cho Patch Model (Lấy feature layer4)
    # Lưu ý: Cần đảm bảo model_patch là model gốc (nếu bọc DataParallel thì phải truy cập .module)
    target_layer = model_patch.module.layer4 if hasattr(model_patch, 'module') else model_patch.layer4
    hook_handle = target_layer.register_forward_hook(get_hook('layer4'))
    
    # 2. Setup Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = BaselineInferDataset(dataroot=dataroot, n_class=n_class, transform=transform)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, batch_size=1)
    
    # Transform riêng cho Patch Model (Cắt Grid)
    trans_patch = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    pred_list = []
    gt_list = []
    
    # Alpha cho SGPS (Có thể truyền qua args)
    alpha = alpha

    for img_name, img_tensor, label_tensor in tqdm(infer_data_loader, desc=f"Inferring SGPS"):
        img_name = img_name[0]
        image_label = label_tensor.to(device) # [1, 4]
        img_tensor = img_tensor.to(device)
        
        orig_img_path = get_image_path(dataroot, img_name)
        
        try:
            orig_img_pil = Image.open(orig_img_path).convert("RGB")
            orig_img_np = np.array(orig_img_pil)
            # Lưu ý: Để chạy Patch Model Grid 7x7, ảnh cần resize về 224x224 (hoặc bội số 32)
            # Ở đây ta resize về 224 để khớp logic training
            img_pil_224 = orig_img_pil.resize((224, 224))
        except Exception as e:
            print(f"Error reading image {orig_img_path}: {e}")
            continue

        # GT Mask (Size gốc)
        gt_map_path_subdir = os.path.join(dataroot, 'mask/', img_name + '.png')
        if os.path.exists(gt_map_path_subdir):
            gt_map = np.array(Image.open(gt_map_path_subdir))
        else:
            gt_map = np.zeros(orig_img_np.shape[:2], dtype=np.uint8)
        gt_list.append(gt_map)

        # --- A. CAM JOINT (TGF-PR) ---
        # Resize input cho CAM về 224 (Input chuẩn của ResNet38)
        input_224 = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            out = model_cam(input_224)
            cam_fused = infer_utils.tri_scale_gated_fusion(out, target_size=(224, 224))
            cam_refined = infer_utils.prototype_refinement(
                cam_fused, out['features'], image_label, target_size=(224, 224), n_class=n_class
            )
            # Lấy Soft Probability của CAM (Masked by Label)
            cam_probs = F.softmax(cam_refined, dim=1) * image_label.view(1, n_class, 1, 1)
            cam_probs_np = cam_probs[0].cpu().numpy() # [4, 224, 224]

        # --- B. PATCH GRID ---
        GRID_SIZE = 7
        PATCH_SIZE = 32
        patches = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                p = img_pil_224.crop((j*PATCH_SIZE, i*PATCH_SIZE, j*PATCH_SIZE+PATCH_SIZE, i*PATCH_SIZE+PATCH_SIZE))
                patches.append(trans_patch(p))
        
        batch_t = torch.stack(patches).to(device)
        
        with torch.no_grad():
            logits = model_patch(batch_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
        grid_pred = preds.reshape(GRID_SIZE, GRID_SIZE)

        # --- C. SGPS REFINEMENT ---
        # Gọi hàm SGPS (trả về mask 224x224)
        refined_224 = self_guided_patch_refinement(beta,
            model_patch,device, img_pil_224, grid_pred, cam_probs_np, n_class=n_class, alpha=alpha
        )
        
        # --- D. UPSAMPLE & POST-PROCESS ---
        # Upsample kết quả SGPS từ 224 lên size gốc
        # Dùng Nearest để giữ giá trị class integer
        refined_224_t = torch.from_numpy(refined_224).unsqueeze(0).unsqueeze(0).float()
        orig_h, orig_w = gt_map.shape
        
        refined_orig = F.interpolate(refined_224_t, size=(orig_h, orig_w), mode='nearest').squeeze().numpy().astype(np.uint8)
        
        # Background Heuristic (Tính trên ảnh gốc)
        # bg_mask = infer_utils.gen_bg_mask(orig_img_np)
        # refined_orig[bg_mask == 1] = 4 
        
        # Protocol: Overwrite BG from GT
        refined_orig[gt_map == 4] = 4
        
        pred_list.append(refined_orig)
        
    # Gỡ hook để tránh leak memory
    hook_handle.remove()
    
    return gt_list, pred_list


def get_mask_baseline(model, dataroot, n_class, args, save_path):
    """
    Hàm tạo Pseudo-mask cho tập Train (Có lọc ảnh sai).
    """
    method = getattr(args, 'method', 'baseline')
    print(f"Generating Pseudo-Masks with Method: [{method.upper()}]")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = BaselineInferDataset(dataroot=dataroot, n_class=n_class, transform=transform)
    loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, batch_size=1)

    skipped_count = 0
    total_count = 0

    for img_name, img_tensor, label_tensor in tqdm(loader, desc="Generating"):
        total_count += 1
        img_name = img_name[0]
        image_label = label_tensor.to(device)
        img_tensor = img_tensor.to(device)
        
        orig_img_path = get_image_path(dataroot, img_name)
        
        try:
            orig_img_pil = Image.open(orig_img_path).convert("RGB")
            orig_img_np = np.array(orig_img_pil)
            orig_img_shape = orig_img_np.shape[:2]
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue

        with torch.no_grad():
            output_dict = model(img_tensor)
            
            # Logic lọc ảnh sai
            features = output_dict['features']
            fc8_weights = model.model.fc8.weight
            raw_cam = F.conv2d(features, fc8_weights)
            logits = torch.mean(raw_cam, dim=(2, 3)) 
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            if not torch.equal(preds, image_label):
                skipped_count += 1
                continue

        if method == 'baseline':
            cam_final = output_dict['cam8']
            cam_upsampled = F.interpolate(cam_final, size=orig_img_shape, mode='bilinear', align_corners=False)[0]
        elif method == 'tgf_pr':
            cam_fused = infer_utils.tri_scale_gated_fusion(output_dict, target_size=orig_img_shape)
            cam_refined = infer_utils.prototype_refinement(
                cam_fused, output_dict['features'], image_label, target_size=orig_img_shape, n_class=n_class
            )
            cam_upsampled = cam_refined[0]

        cam_scores = cam_upsampled.cpu().numpy() * image_label.cpu().clone().view(n_class, 1, 1).numpy()
        pred_mask = np.argmax(cam_scores, axis=0).astype(np.uint8)

        bg_mask = infer_utils.gen_bg_mask(orig_img_np)
        pred_mask[bg_mask == 1] = 4 

        save_file = os.path.join(save_path, img_name + '.png')
        Image.fromarray(pred_mask).save(save_file)
        
    print(f"\n--- GENERATION FINISHED ---")
    print(f"Total: {total_count} | Skipped: {skipped_count} | Kept: {total_count - skipped_count}")