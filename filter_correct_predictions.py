import argparse
import importlib
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import các module cần thiết
# SỬ DỤNG LẠI DATASET GỐC CỦA TÁC GIẢ
from tool.GenDataset import Stage1_TrainDataset
from torchvision import transforms

def filter_predictions(args):
    """
    Duyệt qua tập train, thực hiện phân loại và lọc ra những ảnh
    có dự đoán chính xác tất cả các lớp có mặt, sử dụng label từ tên file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo Model
    # Sử dụng model 'Net' để lấy output phân loại
    model_module = importlib.import_module(args.network)
    model = getattr(model_module, 'Net')(n_class=args.n_class)
    
    # 2. Load Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Load Dataset
    # SỬ DỤNG Stage1_TrainDataset VÌ NÓ ĐỌC LABEL TỪ TÊN FILE
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform, dataset=args.dataset)
    
    # Sử dụng DataLoader để xử lý hiệu quả hơn
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"\nBắt đầu lọc trên {len(dataset)} ảnh từ tập train...")

    correctly_predicted_files = []
    total_images = 0
    correct_count = 0

    # 4. Duyệt qua toàn bộ dataset bằng DataLoader
    for img_names, imgs, ground_truth_labels in tqdm(data_loader, desc="Đang phân loại ảnh"):
        
        # Chuyển ảnh lên GPU
        imgs = imgs.to(device)

        # 5. Thực hiện Inference
        with torch.no_grad():
            x1, x2, x_logits, feature, y_sigmoid = model(imgs)
        
        # 6. Lấy dự đoán từ output của model
        predicted_probs = y_sigmoid.cpu().numpy()
        predicted_labels = (predicted_probs > args.threshold).astype(int)
        
        # 7. So sánh dự đoán với ground truth label
        gt_labels_np = ground_truth_labels.numpy().astype(int)
        
        # So sánh từng ảnh trong batch
        for i in range(len(img_names)):
            total_images += 1
            if np.array_equal(predicted_labels[i], gt_labels_np[i]):
                correct_count += 1
                correctly_predicted_files.append(img_names[i] + '.png')

    # 8. In kết quả và lưu danh sách file
    print("\n" + "="*50)
    print("         KẾT QUẢ LỌC DỰ ĐOÁN")
    print("="*50)
    print(f"Tổng số ảnh đã xử lý: {total_images}")
    print(f"Số ảnh được dự đoán chính xác tất cả các lớp: {correct_count}")
    
    if total_images > 0:
        accuracy_rate = (correct_count / total_images) * 100
        print(f"Tỷ lệ dự đoán chính xác (Exact Match Ratio): {accuracy_rate:.2f}%")

    # 9. Lưu danh sách file vào file text
    output_filename = f"correct_predictions_{args.dataset}.txt"
    with open(output_filename, 'w') as f:
        for filename in correctly_predicted_files:
            f.write(f"{filename}\n")
            
    print(f"\nĐã lưu danh sách {correct_count} file vào: {output_filename}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lọc các ảnh có dự đoán phân loại chính xác")
    
    # Các tham số cần thiết
    parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str, 
                        help="Module python định nghĩa kiến trúc model (phải là 'Net', không phải 'Net_CAM').")
    parser.add_argument("--n_class", default=4, type=int, help="Số lớp foreground.")
    parser.add_argument("--checkpoint_path", default='checkpoints/stage1_baseline_luad.pth', type=str)
    # Sửa lại trainroot để trỏ vào thư mục img/ theo yêu cầu của Stage1_TrainDataset
    parser.add_argument("--trainroot", default='LUAD-HistoSeg/training/', type=str, 
                        help="Thư mục chứa dữ liệu train (thư mục img/).")
    parser.add_argument("--dataset", default="luad", type=str)
    parser.add_argument("--threshold", default=0.5, type=float, 
                        help="Ngưỡng để quyết định một lớp có mặt hay không.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size cho inference.")
    parser.add_argument("--num_workers", default=8, type=int)

    args = parser.parse_args()
    
    filter_predictions(args)