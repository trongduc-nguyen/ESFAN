import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import importlib
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import sys
import cv2
from torch.utils.data import WeightedRandomSampler

# Import modules
sys.path.append('.')
from tool import pyutils, torchutils, iouutils, infer_utils
from tool.GenDataset import Stage1_TrainDataset
from tool.baseline_dataset import BaselineInferDataset
from network.patch_model import PatchModel
from tool.losses import JointTrainingLosses
from PIL import Image

# --- CONFIG ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PALETTE = [[205, 51, 51], [0, 255, 0], [65, 105, 225], [255, 165, 0], [255, 255, 255]] # Class 4: White (Background)

# --- HOOK STORAGE ---
patch_feature_storage = {}
def get_patch_hook(name):
    def hook(model, input, output): patch_feature_storage[name] = output
    return hook

# --- HELPER FUNCTIONS ---
def colorize_mask(mask, n_class):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n_class + 1): # +1 cho class Background
        if i < len(PALETTE): 
            color_mask[mask == i] = PALETTE[i]
    return color_mask

def plot_confidence_distribution(conf_list, epoch, save_dir):
    if len(conf_list) == 0: return
    plt.figure(figsize=(10, 5))
    plt.hist(conf_list, bins=50, range=(0, 1), alpha=0.75, color='blue', edgecolor='black')
    plt.title(f"Confidence Distribution (Epoch {epoch})")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(save_dir, f'conf_dist_epoch_{epoch}.png'))
    plt.close()

def calculate_auto_weights(dataset, n_class, device):
    """
    Tự động tính trọng số dựa trên thống kê tần suất xuất hiện của từng class trong tập train.
    """
    print(">>> Calculating Class Weights (Auto-Balancing)...")
    try:
        if hasattr(dataset, 'labels'):
            labels_np = np.array(dataset.labels)
        elif hasattr(dataset, 'list_labels'):
             labels_np = np.array(dataset.list_labels)
        else:
            print("   Iterating dataset to count labels...")
            labels_list = []
            for i in tqdm(range(len(dataset)), desc="Scanning Labels"):
                _, _, label = dataset[i] 
                labels_list.append(label.numpy())
            labels_np = np.array(labels_list)

        pos_counts = np.sum(labels_np, axis=0)
        total_samples = len(labels_np)
        
        # [LOGIC MỚI]: Inverse Class Frequency
        weights = np.sqrt(total_samples / (pos_counts + 1e-6))
        weights = weights / np.min(weights) # Normalize min=1
        weights = np.clip(weights, 1.0, 5.0) 
        
        print(f"   Counts: {pos_counts.astype(int)} / {total_samples}")
        print(f"   Final Weights: {np.round(weights, 4)}")
        
        return torch.FloatTensor(weights).to(device)
        
    except Exception as e:
        print(f"!! Warning: Could not calculate weights automatically: {e}")
        print("   -> Fallback to default weights [1.0, ...]")
        return torch.ones(n_class).to(device)

def get_fixed_samples(root_dir, n=10, ext='.png'):
    if "val" in root_dir.lower() or "test" in root_dir.lower():
        img_dir = os.path.join(root_dir, 'img')
    else:
        img_dir = root_dir # Trường hợp train folder trực tiếp chứa ảnh
    
    if not os.path.exists(img_dir): # Fallback nếu đường dẫn sai
        return []
        
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(ext)])
    if len(files) > n:
        random.shuffle(files)
        return files[:n]
    return files

def save_visualization(model_s1, model_patch, train_samples, val_samples, args, epoch, device):
    save_dir = os.path.join(args.save_folder, 'visualization', f'epoch_{epoch}')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    model_s1.eval(); model_patch.eval()
    
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )
    
    samples_to_viz = []
    for name in train_samples: samples_to_viz.append({'name': name, 'root': args.trainroot, 'is_train': True})
    for name in val_samples: samples_to_viz.append({'name': name, 'root': args.valroot, 'is_train': False})
        
    trans_img = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    for item in samples_to_viz:
        try:
            if "val" in item['root'].lower():
                img_path = os.path.join(item['root'], 'img', item['name'])
            else:
                img_path = os.path.join(item['root'], 'img', item['name']) # Sửa lại cho đúng cấu trúc train

            # Check path exists
            if not os.path.exists(img_path): continue

            pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
            
            gt_vis = None
            label_tensor = torch.zeros(args.n_class).to(device)
            
            if not item['is_train']:
                mask_root = item['root'].replace('img', 'mask') if 'img' in item['root'] else os.path.join(item['root'], 'mask')
                mask_path = os.path.join(mask_root, item['name'])
                
                if os.path.exists(mask_path):
                    gt_mask = np.array(Image.open(mask_path).resize((224, 224), Image.NEAREST))
                    gt_vis = colorize_mask(gt_mask, args.n_class)
                    present = np.unique(gt_mask)
                    for c in present:
                        if c < args.n_class: label_tensor[c] = 1.0
            else:
                 label_tensor = torch.ones(args.n_class).to(device) # Giả định train có mọi class để viz hết

            img_t = trans_img(pil_img).unsqueeze(0).to(device)
            B = img_t.size(0) 
            
            patches = F.unfold(img_t, kernel_size=32, stride=32).permute(0, 2, 1).contiguous().view(B*49, 3, 32, 32)
            
            with torch.no_grad():
                patch_logits, _, patch_attn_feats = model_patch(patches, batch_size_img=B)
                
                # Forward S1 + FUSION
                # Gọi vào model con bên trong Net_CAM
                out_tuple = model_s1.model(img_t, 
                                           patch_features_seq=patch_attn_feats, 
                                           patch_logits=patch_logits, 
                                           return_features=True)
                
                # --- LOGIC TGF-PR ---
                cam1 = F.relu(model_s1.model.ic1(out_tuple[6])) 
                cam8 = F.relu(model_s1.model.fc8(out_tuple[5])) 
                cam2 = cam8 
                
                output_dict = {
                    'cam1': cam1, 'cam2': cam2, 'cam8': cam8,
                    'features': out_tuple[6] 
                }
                
                cam_fused = infer_utils.tri_scale_gated_fusion(output_dict, target_size=(224, 224))
                
                cam_refined = infer_utils.prototype_refinement(
                    cam_fused, output_dict['features'], label_tensor.unsqueeze(0), (224, 224), args.n_class
                )[0]
                
                cam_scores = cam_refined.cpu().numpy() * label_tensor.view(args.n_class, 1, 1).cpu().numpy()
                cam_pred = np.argmax(cam_scores, axis=0)
                
                # Patch Grid Visual
                preds = torch.argmax(patch_logits, dim=1).cpu().numpy()
                grid_vis = cv2.resize(preds.reshape(7, 7).astype('float32'), (224, 224), interpolation=cv2.INTER_NEAREST).astype('uint8')

            # Protocol BG cho visualization
            if gt_vis is not None:
                 cam_pred[gt_mask == 4] = 4 
            
            # Plot
            cols = 4
            fig, ax = plt.subplots(1, cols, figsize=(20, 5))
            
            img_show = inv_normalize(img_t[0]).cpu().permute(1, 2, 0).numpy()
            img_show = np.clip(img_show, 0, 1)
            
            ax[0].imshow(img_show); ax[0].set_title(f"Input: {item['name']}")
            ax[1].imshow(colorize_mask(grid_vis, args.n_class)); ax[1].set_title("Patch Grid (Local)")
            ax[2].imshow(colorize_mask(cam_pred, args.n_class)); ax[2].set_title("S1 Fused + TGF-PR")
            
            if gt_vis is not None:
                ax[3].imshow(gt_vis); ax[3].set_title("Ground Truth")
            else:
                ax[3].axis('off')
                
            for a in ax: a.axis('off')
            plt.savefig(os.path.join(save_dir, f"{'TRAIN' if item['is_train'] else 'VAL'}_{item['name']}"), bbox_inches='tight')
            plt.close()
            
        except Exception as e: 
            # print(f"Viz Fail {item['name']}: {e}")
            pass
def run_validation(model_s1, model_patch, args, device, phase='val'):
    """
    Hàm Validation/Test chuẩn:
    1. Check kỹ đường dẫn Mask.
    2. Tránh crash nếu không tìm thấy ảnh.
    3. Áp dụng quy tắc Mask Background (Class 4) để tính điểm công bằng.
    """
    # 1. Xác định thư mục dữ liệu
    if phase == 'val': 
        root = args.valroot
    else: 
        root = args.testroot
    
    # 2. Chuyển model sang chế độ đánh giá (quan trọng cho BatchRenorm/Dropout)
    model_s1.eval()
    model_patch.eval()
    
    # 3. Setup DataLoader (Batch size = 1 để giả lập inference từng ảnh)
    dataset = BaselineInferDataset(
        root, 
        args.n_class, 
        transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    gt_list = []
    pred_list = []
    processed_count = 0

    print(f"[{phase.upper()}] Running Validation/Test with TGF-PR...")
    
    # Dùng tqdm để hiện thanh tiến trình
    for img_name, img_t, gt_label_vec in tqdm(loader, desc=f"{phase} Loop", leave=False):
        # Lưu ý: Không dùng try-except bao trùm để tránh nuốt lỗi logic
        
        img_t = img_t.to(device)
        img_name = img_name[0] # Lấy tên file từ tuple
        
        # --- [CRITICAL] CHECK ĐƯỜNG DẪN MASK ---
        # Đảm bảo cấu trúc thư mục là: root/mask/filename.png
        mask_path = os.path.join(root, 'mask', img_name + '.png')
        
        if not os.path.exists(mask_path):
            # Chỉ in warning 1 lần để không spam log nếu sai toàn bộ
            if processed_count == 0: 
                print(f"\n[WARN] Không tìm thấy mask tại: {mask_path}")
                print(f"       -> Check lại tham số --{phase}root hoặc đuôi file (.png/.jpg?)")
            continue
            
        # Load Ground Truth Mask và Resize về 224x224 (Nearest để giữ giá trị nguyên)
        gt_mask = np.array(Image.open(mask_path).resize((224, 224), Image.NEAREST))
        
        # Lấy nhãn ảnh (Image-level label) để lọc prototype
        label_tensor = gt_label_vec[0].to(device) 

        # --- FORWARD PASS (TGF-PR) ---
        # 1. Patch Model Forward
        patches = F.unfold(img_t, kernel_size=32, stride=32).permute(0, 2, 1).contiguous().view(1*49, 3, 32, 32)
        
        with torch.no_grad():
            # Lấy feature từ Patch Model
            patch_logits, _, patch_attn_feats = model_patch(patches, batch_size_img=1)
            
            # 2. Global CAM Forward (Inject Patch Feature)
            out_tuple = model_s1.model(
                img_t, 
                patch_features_seq=patch_attn_feats, 
                patch_logits=patch_logits, 
                return_features=True
            )
            
            # 3. Tri-scale Gated Fusion (TGF)
            conv6_feat = out_tuple[5]      # Feature map trước GAP
            b45_fused_feat = out_tuple[6]  # Feature map mid-level
            
            cam1 = F.relu(model_s1.model.ic1(b45_fused_feat))
            cam8 = F.relu(model_s1.model.fc8(conv6_feat))
            # cam2 lấy từ cam8 (như logic cũ của bạn) hoặc từ nhánh khác nếu có
            cam2 = cam8 
            
            output_dict = {
                'cam1': cam1, 
                'cam2': cam2, 
                'cam8': cam8, 
                'features': b45_fused_feat # Dùng feature này để refine
            }
            
            cam_fused = infer_utils.tri_scale_gated_fusion(output_dict, target_size=(224, 224))
            
            # 4. Prototype Refinement (PR)
            # Lưu ý: Hàm này trả về list các cam đã refine, ta lấy [0] vì batch=1
            cam_refined = infer_utils.prototype_refinement(
                cam_fused, 
                output_dict['features'], 
                label_tensor.unsqueeze(0), 
                (224, 224), 
                args.n_class
            )[0]
            
            # 5. Tạo Final Prediction Mask
            # Masking bằng Image-level label để loại bỏ class không có trong ảnh
            cam_scores = cam_refined.cpu().numpy() * label_tensor.view(args.n_class, 1, 1).cpu().numpy()
            pred = np.argmax(cam_scores, axis=0).astype(np.uint8)
            
            # --- [PROTOCOL] MASK BACKGROUND ---
            # Class 4 trong GT là vùng không xác định/background -> Ép dự đoán thành 4 để không tính lỗi
            pred[gt_mask == 4] = 4
        
        # Append kết quả
        gt_list.append(gt_mask)
        pred_list.append(pred)
        processed_count += 1

    # --- TÍNH TOÁN METRIC ---
    # Check nếu không xử lý được ảnh nào (tránh lỗi chia cho 0)
    if processed_count == 0:
        print(f"\n[ERROR] List kết quả rỗng! Không tính được mIoU cho {phase}.")
        print("        -> Khả năng cao là sai đường dẫn Mask hoặc không load được ảnh.")
        return 0.0

    print(f"-> Đã xử lý thành công {processed_count} ảnh. Đang tính mIoU...")
    res = iouutils.scores(gt_list, pred_list, n_class=args.n_class)
    
    return res['Mean IoU']

# ==============================================================================
# MAIN TRAINING
# ==============================================================================

def train_joint(args):
    device = torch.device("cuda")
    if not os.path.exists(args.save_folder): os.makedirs(args.save_folder)
    
    # 1. SETUP LOGGING
    log_csv = os.path.join(args.save_folder, 'training_log.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Total_Loss', 'Loss_S1', 'Loss_P_Cls', 'Loss_P_Con', 
                                'Pos_Rate', 'Neg_Rate', 'Avg_Conf', 'Val_mIoU'])

    fixed_train_samples = get_fixed_samples(args.trainroot, n=10)
    fixed_val_samples = get_fixed_samples(args.valroot, n=10)

    # 2. MODEL INITIALIZATION
    print("--- Initializing Models ---")
    mod = importlib.import_module(args.network)
    model_s1 = getattr(mod, 'Net_CAM')(n_class=args.n_class)
    
    if args.weights:
        print(f"Loading S1 Weights: {args.weights}")
        if args.weights.endswith('.params'):
            import network.resnet38d
            weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
            model_s1.model.backbone.load_state_dict(weights_dict, strict=False)
        else:
            weights_dict = torch.load(args.weights)
            model_s1.model.backbone.load_state_dict(weights_dict, strict=False)
    model_s1.to(device)

    model_patch = PatchModel(n_class=args.n_class, device=device).to(device)
    # Không cần hook layer4 nữa vì PatchModel mới trả về feature luôn

    # 3. OPTIMIZERS
    param_groups = model_s1.model.get_parameter_groups()
    opt_s1 = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr_s1 * 0.1, 'weight_decay': 5e-4}, 
        {'params': param_groups[1], 'lr': 2*args.lr_s1 * 0.1, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr_s1, 'weight_decay': 5e-4}, 
        {'params': param_groups[3], 'lr': 20*args.lr_s1, 'weight_decay': 0}
    ], lr=args.lr_s1, weight_decay=5e-4, max_step=args.max_epoches * 1000)

    opt_patch = torch.optim.AdamW(model_patch.parameters(), lr=args.lr_patch, weight_decay=1e-4)

    # 4. DATA & LOSS
    trans_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=trans_train, dataset=args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    
    auto_weights = calculate_auto_weights(train_dataset, args.n_class, device)
    print(f"   Normalized Weights: {auto_weights}")
    
    criterion_s1 = nn.BCEWithLogitsLoss(pos_weight=auto_weights)
    criterion_joint = JointTrainingLosses(
        n_tissue_class=args.n_class, 
        class_weights=auto_weights 
    )

    print(f"Start Training | Warmup Epochs: {args.warmup_epoch}")
    best_miou = 0.0

    for ep in range(args.max_epoches):
        model_s1.train(); model_patch.train()
        avg_meter = pyutils.AverageMeter('loss_s1', 'loss_p_cls', 'loss_p_con', 'pos_rate', 'neg_rate')
        pbar = tqdm(train_loader, desc=f"Ep {ep}")
        
        is_warmup = ep < args.warmup_epoch
        # Threshold: Tăng dần từ 0.3 lên max 0.8
        threshold = min(args.threshold + 0.05 * ep , 0.8) 
        
        print(f"-> Epoch {ep} | Warmup: {is_warmup} | Patch Pos Threshold: {threshold:.3f}")
        epoch_confidences = [] 

        for _, imgs, labels in pbar:
            imgs = imgs.to(device); labels = labels.to(device)
            B = imgs.size(0)
            
            opt_s1.zero_grad(); opt_patch.zero_grad()
            
            # --- A. PATCH FORWARD ---
            patches = F.unfold(imgs, kernel_size=32, stride=32).permute(0, 2, 1).contiguous().view(B*49, 3, 32, 32)
            
            # Truyền B vào để Transformer biết đường reshape
            p_logits, p_embeds, p_attn_feats = model_patch(patches, batch_size_img=B) 
            
            # Lưu confidence để thống kê
            with torch.no_grad():
                probs = F.softmax(p_logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                # Chỉ lấy các patch < n_class (Tissue) để thống kê
                tissue_mask = preds < args.n_class
                if tissue_mask.sum() > 0:
                    epoch_confidences.extend(max_probs[tissue_mask].cpu().tolist())

            # --- B. S1 FORWARD (FUSION) ---
            if is_warmup:
                x1, x2, x3, _, _ = model_s1.model(imgs, patch_features_seq=None, patch_logits=None)
            else:
                x1, x2, x3, _, _ = model_s1.model(imgs, patch_features_seq=p_attn_feats, patch_logits=p_logits)

            # --- C. LOSS ---
            loss_s1 = 0.2*criterion_s1(x1, labels) + 0.3*criterion_s1(x2, labels) + 0.5*criterion_s1(x3, labels)
            
            # Loss Patch
            loss_p_cls, stats = criterion_joint.patch_classification_loss(p_logits, labels, thresholds=threshold)
            
            # Loss Contrastive
            loss_p_con = torch.tensor(0.0).to(device)
            if not is_warmup:
                loss_p_con = criterion_joint.patch_contrastive_loss(p_embeds, p_logits)

            if is_warmup:
                total_loss = loss_s1 + loss_p_cls
            else:
                total_loss = loss_s1 + 0.5 * loss_p_cls + 0.1 * loss_p_con
            
            total_loss.backward()
            opt_s1.step(); opt_patch.step()
            
            avg_meter.add({
                'loss_s1': loss_s1.item(), 
                'loss_p_cls': loss_p_cls.item(),
                'loss_p_con': loss_p_con.item(),
                'pos_rate': stats['pos_cnt'] / stats['total_patch'],
                'neg_rate': stats['neg_cnt'] / stats['total_patch']
            })
            
            pbar.set_postfix({'S1': f"{avg_meter.get('loss_s1'):.3f}", 
                              'P_Cls': f"{avg_meter.get('loss_p_cls'):.3f}",
                              'PosR': f"{avg_meter.get('pos_rate'):.2f}"})

        # --- VALIDATION & SAVING ---
        val_miou = run_validation(model_s1, model_patch, args, device, phase='val')
        avg_conf = np.mean(epoch_confidences) if len(epoch_confidences) > 0 else 0.0
        print(f"\n>> Epoch {ep} Summary: Val mIoU (TGF-PR) = {val_miou:.4f} | Avg Conf = {avg_conf:.3f}")
        
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, total_loss.item(), avg_meter.get('loss_s1'), 
                                    avg_meter.get('loss_p_cls'), avg_meter.get('loss_p_con'),
                                    avg_meter.get('pos_rate'), avg_meter.get('neg_rate'),
                                    avg_conf, val_miou])
        
        save_visualization(model_s1, model_patch, fixed_train_samples, fixed_val_samples, args, ep, device)
        plot_confidence_distribution(epoch_confidences, ep, os.path.join(args.save_folder, 'visualization'))
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model_s1.state_dict(), os.path.join(args.save_folder, f'{args.dataset}_joint_s1_best.pth'))
            torch.save(model_patch.state_dict(), os.path.join(args.save_folder, f'{args.dataset}_joint_patch_best.pth'))
            print("-> Saved Best Model.")

    print("Finished Training.")

    # --- FINAL TEST ---
    print("\n" + "="*30)
    print("START EVALUATION ON TEST SET")
    print("="*30)
    
    # Test Final Model
    print(">> [TEST] Evaluating FINAL Model (Last Epoch)...")
    final_test_miou = run_validation(model_s1, model_patch, args, device, phase='test')
    print(f"-> Final Model Test mIoU (TGF-PR): {final_test_miou:.4f}")

    # Test Best Model
    best_s1_path = os.path.join(args.save_folder, f'{args.dataset}_joint_s1_best.pth')
    best_patch_path = os.path.join(args.save_folder, f'{args.dataset}_joint_patch_best.pth')
    
    if os.path.exists(best_s1_path) and os.path.exists(best_patch_path):
        print(f"\n>> [TEST] Loading BEST Model...")
        state = torch.load(best_s1_path, map_location=device)
        model_s1.model.load_state_dict(state, strict=True)
        model_patch.load_state_dict(torch.load(best_patch_path, map_location=device))
        
        print(">> [TEST] Evaluating BEST Model...")
        best_test_miou = run_validation(model_s1, model_patch, args, device, phase='test')
        print(f"-> Best Model Test mIoU (TGF-PR): {best_test_miou:.4f}")
    else:
        best_test_miou = 0.0

    # Save Test Report
    report_file = os.path.join(args.save_folder, 'test_result_report.txt')
    with open(report_file, 'w') as f:
        f.write("=== TEST SET EVALUATION REPORT ===\n")
        f.write(f"Protocol: Standard (TGF-PR Refinement)\n")
        f.write(f"Final Model mIoU: {final_test_miou:.4f}\n")
        f.write(f"Best Model mIoU : {best_test_miou:.4f}\n")
    
    print(f"\nTest results saved to: {report_file}")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str)
    
    parser.add_argument("--lr_s1", default=0.01, type=float) 
    parser.add_argument("--lr_patch", default=0.001, type=float)
    parser.add_argument("--threshold", default=0.3, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    
    # Path Arguments
    parser.add_argument("--trainroot", default='LUAD-HistoSeg/training/', type=str)
    parser.add_argument("--valroot", default='LUAD-HistoSeg/val/', type=str)
    parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str)
    
    parser.add_argument("--save_folder", default='checkpoints_joint_v2', type=str)
    parser.add_argument("--dataset", default='luad', type=str)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    
    args = parser.parse_args()
    train_joint(args)