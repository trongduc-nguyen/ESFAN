import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import cv2

# ===================================================================================
# IMPORT VÀ ĐỊNH NGHĨA MÔ HÌNH
# ===================================================================================
try:
    from network.resnet38d_baseline import Net as BaselineBackbone
except ImportError:
    print("LỖI: Không thể import từ 'network/resnet38d_baseline.py'.")
    print("Hãy chắc chắn bạn đã tạo file này và đặt nó đúng trong thư mục 'network/'.")
    exit()

class Net(torch.nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.backbone = BaselineBackbone()
        self.classifier = torch.nn.Linear(4096, n_class)
    def forward(self, x):
        _, _, feature_map = self.backbone(x)
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).view(feature_map.size(0), -1)
        logits = self.classifier(pooled)
        return logits
    def forward_cam(self, x):
        _, _, feature_map = self.backbone(x)
        return feature_map

# ===================================================================================
# HÀM HỖ TRỢ VÀ LOGIC CHÍNH
# ===================================================================================

def get_model(weights_path, n_class):
    """Khởi tạo mô hình và tải trọng số đã huấn luyện."""
    model = Net(n_class=n_class)
    try:
        model.load_state_dict(torch.load(weights_path))
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file checkpoint tại '{weights_path}'")
        exit()
    model = model.cuda().eval()
    print(f"Đã tải thành công model từ: {weights_path}")
    return model

def find_background_mask(image_path, threshold=220):
    """
    Xác định vùng background trắng xóa từ ảnh gốc bằng OpenCV.
    """
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Cảnh báo: Không thể đọc ảnh {image_path}")
        return None
        
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Phân ngưỡng để lấy vùng màu trắng/rất sáng
    _, bg_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    
    # Dùng các phép toán hình thái học để loại bỏ nhiễu và lấp lỗ hổng
    kernel = np.ones((5, 5), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Trả về mask dạng boolean (True nếu là background)
    return bg_mask == 255

def create_and_save_labels(weights, img_dir, save_dir, n_class, bg_threshold, visualize):
    """
    Hàm chính để tạo và lưu các pseudo-label đã được làm sạch.
    """
    visualize = True
    # 1. Chuẩn bị model, thư mục và các thành phần cần thiết
    model = get_model(weights, n_class)
    classifier_weights = model.classifier.weight.data.cpu().numpy()
    transform_infer = transforms.Compose([transforms.ToTensor()])

    try:
        image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"LỖI: Không tìm thấy ảnh nào trong '{img_dir}'")
            return
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy thư mục '{img_dir}'")
        return
        
    # Tạo thư mục lưu chính cho dữ liệu huấn luyện
    os.makedirs(save_dir, exist_ok=True)
    print(f"Các pseudo-label (dùng để training) sẽ được lưu tại: {save_dir}")
    
    # Tạo thư mục con cho ảnh trực quan hóa nếu người dùng yêu cầu
    vis_save_dir = None
    if True:
        vis_save_dir = os.path.join(save_dir, "visualizable")
        os.makedirs(vis_save_dir, exist_ok=True)
        print(f"Các pseudo-label (để nhìn) sẽ được lưu tại: {vis_save_dir}")

    # 2. Lặp qua tất cả ảnh trong thư mục
    for filename in tqdm(image_files, desc="Đang tạo pseudo-labels"):
        img_path = os.path.join(img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform_infer(image).unsqueeze(0).cuda()

        with torch.no_grad():
            feature_map_tensor = model.forward_cam(img_tensor)
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)[0]
            predicted_labels = torch.where(probs > 0.5)[0].cpu().numpy()
            
            if len(predicted_labels) == 0:
                final_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                # Tạo pseudo-mask từ CAM (logic mượt mà)
                cams_raw = (classifier_weights[predicted_labels] @ feature_map_tensor.cpu().numpy()[0].reshape(feature_map_tensor.shape[1], -1))
                cams_raw = cams_raw.reshape(len(predicted_labels), *feature_map_tensor.shape[2:])
                
                normalized_cams = []
                for cam in cams_raw:
                    cam_resized = F.interpolate(torch.from_numpy(cam[np.newaxis, np.newaxis, :, :]), size=image.size[::-1], mode='bilinear', align_corners=False)[0, 0]
                    cam_min, cam_max = cam_resized.min(), cam_resized.max()
                    if cam_max > cam_min:
                        normalized_cams.append((cam_resized - cam_min) / (cam_max - cam_min))
                    else:
                        normalized_cams.append(torch.zeros_like(cam_resized))
                
                if not normalized_cams:
                    pseudo_mask_cam = np.zeros((image.height, image.width), dtype=np.uint8)
                else:
                    normalized_cams_tensor = torch.stack(normalized_cams, dim=0)
                    _, pseudo_mask_indices = torch.max(normalized_cams_tensor, dim=0)
                    pseudo_mask_cam = predicted_labels[pseudo_mask_indices.numpy()]
                
                # Xử lý background và đè lên mask
                background_mask = find_background_mask(img_path, threshold=bg_threshold)
                
                final_mask = pseudo_mask_cam.copy()
                if background_mask is not None:
                    final_mask[background_mask] = 0
        
        # In ra các giá trị nhãn duy nhất có trong mask cuối cùng
        unique_labels = np.unique(final_mask)
        tqdm.write(f"Ảnh: {filename} -> Các nhãn được tạo: {unique_labels}")

        # --- Lưu kết quả ---
        
        # 1. Lưu mask nguyên bản (giá trị 0,1,2,3,...) để training
        mask_image_train = Image.fromarray(final_mask.astype(np.uint8))
        mask_image_train.save(os.path.join(save_dir, filename))

        # 2. Nếu có cờ --visualize, lưu thêm một phiên bản đã scale để dễ nhìn
        if visualize:
            scaling_factor = 50 
            visual_mask_array = (final_mask * scaling_factor).astype(np.uint8)
            
            mask_image_vis = Image.fromarray(visual_mask_array)
            mask_image_vis.save(os.path.join(vis_save_dir, filename))

    print("\n--- Hoàn tất! Tất cả pseudo-label đã được tạo và lưu. ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Giai đoạn 1: Tạo và làm sạch Pseudo-Labels.")
    
    parser.add_argument("--weights", type=str, required=True,
                        help="Đường dẫn đến file checkpoint đã huấn luyện (ví dụ: checkpoints/stage1_baseline_cam_on_luad.pth).")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Đường dẫn đến thư mục chứa ảnh huấn luyện (ví dụ: ./LUAD-HistoSeg/training/).")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Thư mục để lưu các pseudo-label đã được làm sạch.")
    parser.add_argument("--n_class", type=int, required=True,
                        help="Số lượng class của dataset (ví dụ: 4).")
    parser.add_argument("--bg_threshold", type=int, default=220,
                        help="Ngưỡng độ sáng để xác định background (pixel > threshold là background).")
    
    # Thêm cờ mới để bật/tắt chế độ trực quan hóa
    parser.add_argument("--visualize", action='store_true', 
                        help="Tạo thêm một bộ pseudo-label đã scale giá trị để dễ dàng xem bằng mắt thường.")
    
    args = parser.parse_args()
    
    create_and_save_labels(
        weights=args.weights,
        img_dir=args.img_dir,
        save_dir=args.save_dir,
        n_class=args.n_class,
        bg_threshold=args.bg_threshold,
        visualize=args.visualize
    )