import argparse
import importlib
import torch
import os
import json
import numpy as np
from tool import infer_fun_baseline
from tool import iouutils

def check_missing_classes(gt_list, pred_list, n_class):
    """
    Kiểm tra số lượng ảnh bị dự đoán thiếu lớp so với ground truth.
    In ra tổng số trường hợp và tỷ lệ.
    """
    missing_class_cases = 0
    total_images = len(gt_list)
    
    print("\n--- Bắt đầu kiểm tra các trường hợp bị thiếu lớp ---")
    
    for i in range(total_images):
        gt_mask = gt_list[i]
        pred_mask = pred_list[i]
        
        # Lấy danh sách các lớp foreground (0, 1, 2, 3) có trong ground truth
        gt_classes = set(np.unique(gt_mask)) - {4} # Loại bỏ lớp 4 (background)
        
        # Lấy danh sách các lớp foreground được model dự đoán
        pred_classes = set(np.unique(pred_mask)) - {4} # Loại bỏ lớp 4 (background)
        
        # Kiểm tra xem có lớp nào trong ground truth mà không có trong dự đoán không
        if not pred_classes.issuperset(gt_classes):
            missing_classes = gt_classes - pred_classes
            missing_class_cases += 1
            # Bỏ comment dòng dưới nếu bạn muốn xem chi tiết từng ảnh bị lỗi
            # print(f"  - Ảnh {i}: Bị thiếu lớp {missing_classes}. GT có {gt_classes}, Pred có {pred_classes}")

    print("\n--- KẾT QUẢ KIỂM TRA THIẾU LỚP ---")
    print(f"Tổng số ảnh trong tập test: {total_images}")
    print(f"Số trường hợp ảnh bị dự đoán thiếu lớp: {missing_class_cases}")
    
    if total_images > 0:
        miss_rate = (missing_class_cases / total_images) * 100
        print(f"Tỷ lệ ảnh bị thiếu lớp: {miss_rate:.2f}%")
    print("=" * 35)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str)
    parser.add_argument("--n_class", default=4, type=int, help="Số lớp foreground.")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--dataset", default="luad", type=str)
    
    # Paths
    parser.add_argument("--checkpoint_path", default='checkpoints/stage1_baseline_luad.pth', type=str)
    parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str, help="Thư mục chứa dữ liệu test (img và mask)")
    parser.add_argument("--trainroot", default='LUAD-HistoSeg/training/', type=str, help="Thư mục chứa dữ liệu train (để tạo mask)")
    parser.add_argument("--mask_save_path", default='results/baseline_pseudo_masks/', type=str)

    # Action
    parser.add_argument("--action", choices=['evaluate', 'generate_masks'], required=True, help="Chọn 'evaluate' để tính score trên tập test, hoặc 'generate_masks' để tạo pseudo-mask trên tập train.")
    
    args = parser.parse_args()

    # Xử lý Device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo model
    model_module = importlib.import_module(args.network)
    model = getattr(model_module, 'Net_CAM')(n_class=args.n_class)
    
    # 2. Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    
    # 3. Chuyển model lên device và đặt ở chế độ eval
    model.to(device)
    model.eval()

    if args.action == 'evaluate':
        print("\n--- Bắt đầu đánh giá trên tập Test (sử dụng iouutils.py gốc) ---")
        
        # 1. Lấy danh sách GT và dự đoán
        gt_list, pred_list = infer_fun_baseline.infer_baseline(model, args.testroot, args.n_class, args)
        
        # 2. Gọi hàm scores để tính các chỉ số chính
        results = iouutils.scores(gt_list, pred_list, n_class=args.n_class)
        
        print("\n--- KẾT QUẢ ĐÁNH GIÁ BASELINE ---")
        print(json.dumps(results, indent=4))
        print("=" * 35)
        
        # 3. Gọi hàm kiểm tra thiếu lớp
        check_missing_classes(gt_list, pred_list, args.n_class)

    elif args.action == 'generate_masks':
        print(f"\n--- Bắt đầu tạo Pseudo-Masks từ tập Train ---")
        print(f"Masks sẽ được lưu tại: {args.mask_save_path}")
        infer_fun_baseline.get_mask_baseline(model, args.trainroot, args.n_class, args, args.mask_save_path)
        print("\n--- Hoàn tất tạo Pseudo-Masks ---")

if __name__ == '__main__':
    main()

# # File: evaluate_baseline.py (PHIÊN BẢN SO SÁNH)

# import argparse
# import importlib
# import torch
# import os
# import json
# import numpy as np

# # Import các file cần thiết
# from tool import infer_fun_baseline
# from tool import iouutils

# def check_missing_classes(gt_list, pred_list, n_class, method_name=""):
#     # (Hàm này giữ nguyên)
#     missing_class_cases = 0
#     total_images = len(gt_list)
#     for gt_mask, pred_mask in zip(gt_list, pred_list):
#         gt_classes = set(np.unique(gt_mask)) - {4}
#         pred_classes = set(np.unique(pred_mask)) - {4}
#         if not pred_classes.issuperset(gt_classes):
#             missing_class_cases += 1
#     print(f"\n--- KẾT QUẢ KIỂM TRA THIẾU LỚP ({method_name}) ---")
#     print(f"Tổng số ảnh trong tập test: {total_images}")
#     print(f"Số trường hợp ảnh bị dự đoán thiếu lớp: {missing_class_cases}")
#     if total_images > 0:
#         miss_rate = (missing_class_cases / total_images) * 100
#         print(f"Tỷ lệ ảnh bị thiếu lớp: {miss_rate:.2f}%")
#     print("=" * 45)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str)
#     parser.add_argument("--n_class", default=4, type=int, help="Số lớp foreground.")
#     parser.add_argument("--num_workers", default=8, type=int)
#     parser.add_argument("--checkpoint_path", default='/home/25duc.nt3/ESFAN/checkpoints_joint/joint_model_epoch_10.pth', type=str)
#     parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str)
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Sử dụng thiết bị: {device}")

#     # --- Load Model (chỉ load 1 lần) ---
#     model_module = importlib.import_module(args.network)
#     model = getattr(model_module, 'Net_CAM')(n_class=args.n_class)
#     print(f"Loading checkpoint from: {args.checkpoint_path}")
#     model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
#     model.to(device)
#     model.eval()

#     # =======================================================================
#     #                       PHẦN ĐÁNH GIÁ
#     # =======================================================================

#     # --- 1. Đánh giá CAM Baseline ---
#     print("\n" + "="*20 + " Bắt đầu đánh giá CAM Baseline " + "="*20)
#     args.method = 'cam' # Đặt "công tắc"
#     # 1a. Lấy danh sách GT và dự đoán
#     gt_list_cam, pred_list_cam = infer_fun_baseline.infer_baseline(model, 1, args.testroot, args.n_class, args)
#     # 1b. Gọi hàm scores để tính các chỉ số chính
#     cam_results = iouutils.scores(gt_list_cam, pred_list_cam, n_class=args.n_class)
    
#     # --- 2. Đánh giá Cosine K-Means ---
#     print("\n" + "="*20 + " Bắt đầu đánh giá Cosine K-Means " + "="*20)
#     args.method = 'cosine' # Đặt "công tắc"
#     # 2a. Lấy danh sách GT và dự đoán
#     gt_list_cosine, pred_list_cosine = infer_fun_baseline.infer_baseline(model, 100,args.testroot, args.n_class, args)
#     # 2b. Gọi hàm scores để tính các chỉ số chính
#     cosine_results = iouutils.scores(gt_list_cosine, pred_list_cosine, n_class=args.n_class)

#     # --- 3. In kết quả tổng hợp ---
#     print("\n\n" + "="*25 + " KẾT QUẢ CUỐI CÙNG " + "="*25)
#     print("\n--- KẾT QUẢ TỪ CAM GỐC ---")
#     print(json.dumps(cam_results, indent=4))
#     print("\n--- KẾT QUẢ TỪ K-MEANS (COSINE) ---")
#     print(json.dumps(cosine_results, indent=4))

#     print("\n--- TÓM TẮT SO SÁNH ---")
#     print(f"{'Method':<25} | {'Mean IoU':<12} | {'Mean Dice':<12}")
#     print("-" * 55)
#     print(f"{'CAM Baseline':<25} | {cam_results['Mean IoU']:.4f}{'':<5} | {cam_results['Mean Dice']:.4f}")
#     print(f"{'K-Means (Cosine)':<25} | {cosine_results['Mean IoU']:.4f}{'':<5} | {cosine_results['Mean Dice']:.4f}")
#     print("=" * 55)

#     # --- 4. Kiểm tra thiếu lớp cho cả hai ---
#     check_missing_classes(gt_list_cam, pred_list_cam, args.n_class, "CAM Baseline")
#     check_missing_classes(gt_list_cosine, pred_list_cosine, args.n_class, "K-Means (Cosine)")


# if __name__ == '__main__':
#     main()