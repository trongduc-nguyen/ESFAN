import argparse
import importlib
import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from tool.baseline_dataset import BaselineInferDataset
from tool.iouutils import _fast_hist
from tool.infer_utils import gen_bg_mask
from torchvision import transforms

def calculate_metrics_from_lists(gt_list, pred_list, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(gt_list, pred_list):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    iou[np.isnan(iou)] = 0
    return iou

def compare_logic(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Load Model
    model_module = importlib.import_module(args.network)
    model = getattr(model_module, 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = BaselineInferDataset(dataroot=args.testroot, n_class=args.n_class, transform=transform)
    
    gt_list, pred_list_v1, pred_list_v2 = [], [], []

    print(f"\nBắt đầu so sánh trên {len(dataset)} ảnh từ tập test...")

    # 4. Duyệt qua toàn bộ dataset
    for i in tqdm(range(len(dataset)), desc="Đang xử lý ảnh"):
        img_name, img, label = dataset[i]
        
        img_path = os.path.join(args.testroot, 'img/', img_name + '.png')
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        orig_img_size = orig_img.shape[:2]
        
        gt_map_path = os.path.join(args.testroot, 'mask/', img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)

        # Lấy CAM từ model
        with torch.no_grad():
            cam = model(img.unsqueeze(0).to(device))
            cam = F.interpolate(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
            cam_scores_raw = (cam.cpu().numpy() * label.clone().view(args.n_class, 1, 1).numpy()).copy()

        # --- TÍNH TOÁN VÀ LƯU KẾT QUẢ CHO CẢ 2 LOGIC ---
        
        # Logic V1 (So sánh điểm - CÓ CHUẨN HÓA)
        # Đây là logic cho ra ~74% mIoU
        cam_v1 = cam_scores_raw.copy()
        cam_v1[cam_v1 < 0] = 0 # Đảm bảo không âm
        # BƯỚC CHUẨN HÓA BỊ THIẾU
        max_val_v1 = np.max(cam_v1)
        if max_val_v1 > 1e-8:
            cam_v1 = cam_v1 / max_val_v1
        
        bg_score_v1 = np.expand_dims(gen_bg_mask(orig_img), axis=0)
        bgcam_score_v1 = np.concatenate((cam_v1, bg_score_v1), axis=0)
        seg_map_v1 = np.argmax(bgcam_score_v1, axis=0)
        pred_list_v1.append(seg_map_v1)
        
        # Logic V2 (Ghi đè - KHÔNG CẦN CHUẨN HÓA)
        # Đây là logic cho ra ~72% mIoU
        cam_v2 = cam_scores_raw.copy()
        foreground_mask_v2 = np.argmax(cam_v2, axis=0)
        bg_binary_mask_v2 = (gen_bg_mask(orig_img) > 0)
        seg_map_v2 = foreground_mask_v2.copy()
        seg_map_v2[bg_binary_mask_v2] = 4
        pred_list_v2.append(seg_map_v2)
        
    # 5. Tính toán và in kết quả
    num_total_classes = args.n_class + 1
    iou_v1 = calculate_metrics_from_lists(gt_list, pred_list_v1, num_total_classes)
    iou_v2 = calculate_metrics_from_lists(gt_list, pred_list_v2, num_total_classes)

    bg_iou_v1 = iou_v1[4]
    bg_iou_v2 = iou_v2[4]
    
    fg_miou_v1 = np.nanmean(iou_v1[:4])
    fg_miou_v2 = np.nanmean(iou_v2[:4])

    print("\n" + "="*50)
    print("         KẾT QUẢ SO SÁNH HIỆU SUẤT BACKGROUND")
    print("="*50)
    print(f"{'Chỉ số':<30} | {'Logic CŨ (So sánh điểm)':<25} | {'Logic MỚI (Ghi đè)':<20}")
    print("-"*80)
    print(f"{'IoU của Background (Lớp 4)':<30} | {bg_iou_v1:^25.4f} | {bg_iou_v2:^20.4f}")
    print(f"{'mIoU của Foreground (Lớp 0-3)':<30} | {fg_miou_v1:^25.4f} | {fg_miou_v2:^20.4f}")
    print("="*80)


    if bg_iou_v2 > bg_iou_v1:
        print("\nNhận xét: Logic MỚI (Ghi đè) tạo ra background TỐT HƠN một cách đáng kể.")
    else:
        print("\nNhận xét: Logic CŨ (So sánh điểm) tạo ra background tốt hơn hoặc tương đương.")
        
    if fg_miou_v2 < fg_miou_v1:
         print("Tuy nhiên, điều này phải trả giá bằng việc làm giảm hiệu suất trên các lớp foreground.")
    print("Điều này xác nhận sự đánh đổi (trade-off) giữa việc làm sạch background và giữ lại chi tiết foreground.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="So sánh 2 logic xử lý background")
    parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--checkpoint_path", default='checkpoints/stage1_baseline_luad.pth', type=str)
    parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str)
    args = parser.parse_args()
    
    compare_logic(args)