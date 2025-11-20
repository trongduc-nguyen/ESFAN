# File: prepare_oneshot_data.py (ĐÃ SỬA LỖI LOGIC NGƯỠNG)

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import importlib
import cv2
from skimage import morphology
import shutil

# --- CÁC THAM SỐ ---
ORIG_TRAIN_ROOT = 'LUAD-HistoSeg/training/'
OUTPUT_ROOT = 'LUAD-HistoSeg_OneShot/'
CHECKPOINT_PATH = 'checkpoints/stage1_baseline_luad.pth'
NETWORK_MODULE = 'network.resnet38_cls_baseline'
N_CLASS = 4
# NGƯỠNG TƯƠNG ĐỐI: Lấy 30% các pixel có cường độ kích hoạt cao nhất
RELATIVE_CONF_THRESHOLD = 0.1 

# =======================================================================

def main():
    """Hàm chính thực thi toàn bộ pipeline chuẩn bị dữ liệu."""

    if os.path.exists(OUTPUT_ROOT):
        print(f"Thư mục '{OUTPUT_ROOT}' đã tồn tại. Xóa và tạo lại...")
        shutil.rmtree(OUTPUT_ROOT)
    
    output_image_dir = os.path.join(OUTPUT_ROOT, 'images')
    output_mask_dir = os.path.join(OUTPUT_ROOT, 'instance_masks')
    os.makedirs(output_image_dir)
    os.makedirs(output_mask_dir)
    print(f"Đã tạo các thư mục đầu ra tại: '{OUTPUT_ROOT}'")

    print("Đang tải model phân loại...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_module = importlib.import_module(NETWORK_MODULE)
    cls_model = getattr(model_module, 'Net')(n_class=N_CLASS)
    cam_model = getattr(model_module, 'Net_CAM')(n_class=N_CLASS)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    cls_model.load_state_dict(state_dict)
    cam_model.load_state_dict(state_dict)
    cls_model.to(device).eval()
    cam_model.to(device).eval()
    print("Model đã được tải thành công!")

    all_train_files = []
    for root, _, files in os.walk(ORIG_TRAIN_ROOT):
        for f in files:
            if f.endswith('.png'):
                all_train_files.append(os.path.join(root, f))

    num_filtered_images = 0
    transform = transforms.Compose([transforms.ToTensor()])

    for image_path in tqdm(all_train_files, desc="Chuẩn bị dữ liệu One-Shot"):
        img_name = os.path.basename(image_path)
        
        try:
            label_str = img_name.split(']')[0].split('[')[-1]
            gt_label = torch.Tensor([int(c) for c in label_str.split(' ')])
        except (IndexError, ValueError):
            continue
        
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, _, logits, _, _ = cls_model(img_tensor)
        pred_probs = torch.sigmoid(logits).squeeze(0).cpu()
        pred_label = (pred_probs > 0.5).float()

        if torch.equal(gt_label, pred_label) and gt_label.sum() >= 2:
            
            with torch.no_grad():
                cam_raw = cam_model(img_tensor)

            orig_img_np = np.array(img_pil)
            cam_upsampled = F.interpolate(cam_raw, size=orig_img_np.shape[:2], mode='bilinear', align_corners=False)[0]
            cam_scores = cam_upsampled.cpu().numpy() * gt_label.clone().view(N_CLASS, 1, 1).numpy()
            
            gray = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            binary_mask = (binary == 255)
            true_bg_mask = morphology.remove_small_objects(binary_mask, min_size=50, connectivity=1)

            instance_mask = np.zeros(orig_img_np.shape[:2], dtype=np.uint8)
            instance_id_counter = 1
            
            present_classes_indices = torch.where(gt_label == 1)[0].numpy()
            
            for class_idx in present_classes_indices:
                
                # ***** PHẦN LOGIC NGƯỠNG ĐÃ ĐƯỢC SỬA LẠI TẠI ĐÂY *****
                
                class_cam = cam_scores[class_idx]
                
                # Bỏ qua nếu không có kích hoạt nào
                if class_cam.max() <= 0:
                    continue

                # 1. Chuẩn hóa CAM của lớp này về khoảng [0, 1]
                #    (Min-Max Normalization)
                normalized_cam = (class_cam - class_cam.min()) / (class_cam.max() - class_cam.min())
                
                # 2. Áp dụng ngưỡng tương đối trên CAM đã chuẩn hóa
                binary_support_mask = (normalized_cam > RELATIVE_CONF_THRESHOLD)
                
                # **********************************************************

                # Các bước còn lại giữ nguyên
                binary_support_mask[true_bg_mask] = False
                binary_support_mask = morphology.remove_small_objects(binary_support_mask, min_size=100, connectivity=1)

                if binary_support_mask.sum() > 0:
                    instance_mask[binary_support_mask] = instance_id_counter
                    instance_id_counter += 1
            
            if instance_mask.max() > 0:
                num_filtered_images += 1
                img_pil.save(os.path.join(output_image_dir, img_name))
                mask_pil = Image.fromarray(instance_mask)
                mask_pil.save(os.path.join(output_mask_dir, img_name))

    print("\n" + "="*50)
    print("Hoàn tất chuẩn bị dữ liệu!")
    print(f"Tổng số ảnh ban đầu: {len(all_train_files)}")
    print(f"Số lượng ảnh được lọc và lưu lại: {num_filtered_images}")
    print("="*50)


if __name__ == '__main__':
    main()