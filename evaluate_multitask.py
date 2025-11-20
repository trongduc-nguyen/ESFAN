# evaluate_multitask.py

import argparse
import importlib
import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from skimage import morphology

# Import các module cần thiết
from tool.baseline_dataset import BaselineInferDataset
from tool.iouutils import scores as calculate_scores_from_lists
from torchvision import transforms
from torch.utils.data import DataLoader
def check_missing_classes(gt_list, pred_list, filenames):
    """
    Kiểm tra và in ra tên các file bị dự đoán thiếu lớp.
    """
    missing_class_cases = 0
    total_images = len(gt_list)
    print("\n--- Bắt đầu kiểm tra các trường hợp bị thiếu lớp ---")
    
    for i in range(total_images):
        img_name = filenames[i]
        gt_mask = gt_list[i]
        pred_mask = pred_list[i]
        
        gt_classes = set(np.unique(gt_mask)) - {4}
        pred_classes = set(np.unique(pred_mask)) - {4}
        
        if not pred_classes.issuperset(gt_classes):
            missing_classes = gt_classes - pred_classes
            missing_class_cases += 1
            print(f"  - Lỗi tại ảnh: {img_name}.png")
            print(f"    -> Bị thiếu lớp: {missing_classes}. (GT có {gt_classes}, Pred chỉ có {pred_classes})")

    print("\n--- KẾT QUẢ KIỂM TRA THIẾU LỚP ---")
    print(f"Tổng số ảnh trong tập test: {total_images}")
    print(f"Số trường hợp ảnh bị dự đoán thiếu lớp: {missing_class_cases}")
    if total_images > 0:
        miss_rate = (missing_class_cases / total_images) * 100
        print(f"Tỷ lệ ảnh bị thiếu lớp: {miss_rate:.2f}%")
    print("=" * 35)

def infer_multitask(model, dataroot, n_class, args):
    """
    Thực hiện inference trên tập test sử dụng model MultiTaskNet.
    Sử dụng logic "ghi đè" background để tạo segmentation map cuối cùng.
    """
    model.eval()
    pred_list = []
    gt_list = []
    filenames = []
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = BaselineInferDataset(dataroot=dataroot, n_class=n_class, transform=transform)
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, batch_size=1) 

    for img_name, img, label in tqdm(data_loader, desc="Đang đánh giá"):
        img_name = img_name[0]
        label = label[0]
        filenames.append(img_name)
        
        img_path = os.path.join(dataroot, 'img/', img_name + '.png')
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        orig_img_size = orig_img.shape[:2]

        with torch.no_grad():
            # Cho ảnh đi qua model. Chúng ta chỉ cần CAM.
            _, _, _, cam = model(img.to(model.device))
            cam = F.interpolate(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
            cam_scores = cam.cpu().numpy() * label.clone().view(n_class, 1, 1).numpy()
        
        # --- SỬ DỤNG LOGIC "GHI ĐÈ" ĐÃ ĐƯỢC KIỂM CHỨNG ---
        foreground_mask = np.argmax(cam_scores, axis=0)
        
        gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bg_binary_mask = morphology.remove_small_objects((binary == 255), min_size=50, connectivity=1)

        seg_map = foreground_mask.copy()
        seg_map[bg_binary_mask] = 4 # Ghi đè background
        pred_list.append(seg_map)
        
        gt_map_path = os.path.join(dataroot, 'mask/', img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)
        
    return gt_list, pred_list, filenames

def main():
    parser = argparse.ArgumentParser(description="Đánh giá model Multi-Task đã được huấn luyện")
    
    parser.add_argument("--network", default="network.multitask_model", type=str,
                        help="Module python định nghĩa kiến trúc MultiTaskNet.")
    parser.add_argument("--n_class", default=4, type=int, help="Số lớp foreground.")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--dataset", default="luad", type=str)
    
    # Paths
    parser.add_argument("--checkpoint_path",default='/home/25duc.nt3/ESFAN/checkpoints_mutitask/multitask_epoch_10.pth', type=str,
                        help="Đường dẫn đến checkpoint của model Multi-Task (ví dụ: multitask_epoch_10.pth).")
    parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str,
                        help="Thư mục chứa dữ liệu test (img và mask).")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo model
    model_module = importlib.import_module(args.network)
    model = getattr(model_module, 'MultiTaskNet')(n_class=args.n_class)
    # Thêm một thuộc tính device vào model để dễ truy cập
    model.device = device
    
    # 2. Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    # Load state dict đầy đủ, strict=True vì chúng ta lưu toàn bộ model
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=True)
    
    # 3. Chuyển model lên device và đặt ở chế độ eval
    model.to(device)
    model.eval()

    # 4. Thực hiện inference và lấy kết quả
    print("\n--- Bắt đầu đánh giá trên tập Test ---")
    gt_list, pred_list, filenames = infer_multitask(model, args.testroot, args.n_class, args)
    
    # 5. Gọi hàm scores để tính các chỉ số chính
    results = calculate_scores_from_lists(gt_list, pred_list, n_class=args.n_class)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MODEL MULTI-TASK ---")
    print(json.dumps(results, indent=4))
    print("=" * 35)
    
    # 6. Gọi hàm kiểm tra thiếu lớp
    check_missing_classes(gt_list, pred_list, filenames)

if __name__ == '__main__':
    main()

