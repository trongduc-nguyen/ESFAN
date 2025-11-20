# File: tool/infer_fun_baseline.py (PHIÊN BẢN NÂNG CẤP)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
from tqdm import tqdm

# Import các file cần thiết
from .baseline_dataset import BaselineInferDataset
from . import infer_utils

# =======================================================================
#    CÁC HÀM TIỆN ÍCH CHO COSINE K-MEANS (Được thêm vào file này)
# =======================================================================
def cosine_similarity_pytorch(x, centers):
    x_norm = F.normalize(x, p=2, dim=1)
    centers_norm = F.normalize(centers, p=2, dim=1)
    return torch.matmul(x_norm, centers_norm.t())

def refine_with_cosine_kmeans(model, image_tensor, image_label_tensor, orig_img_shape, cam_threshold_for_init=0.5, n_iters=10):
    device = image_tensor.device
    image_label = image_label_tensor.cpu()
    with torch.no_grad():
        _, _, feature_map = model.model.backbone(image_tensor)
    C, H, W = feature_map.shape[1], feature_map.shape[2], feature_map.shape[3]
    features_reshaped = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, C)
    with torch.no_grad():
        cam_raw = model(image_tensor)
    label_view = image_label.view(len(image_label), 1, 1)
    cam_scores_tensor = cam_raw.cpu() * label_view
    present_classes_indices = torch.where(image_label == 1)[0].numpy()
    k = len(present_classes_indices)
    if k == 0: return np.full(orig_img_shape, 4, dtype=np.int32)
    
    initial_centroids = []
    valid_present_classes = []
    
    cam_resized_for_init = F.interpolate(cam_scores_tensor, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    
    # ==================================================================
    #    PHẦN KHỞI TẠO TÂM CỤM - CHỌN 1 TRONG CÁC PHƯƠNG PHÁP DƯỚI ĐÂY
    # ==================================================================
    for class_idx in present_classes_indices:
        # Lấy bản đồ CAM cho lớp hiện tại
        class_cam_scores = cam_resized_for_init[class_idx]
        
        # Mask cho các vùng có độ tin cậy cao
        high_conf_mask = (class_cam_scores > cam_threshold_for_init)
        
        if high_conf_mask.sum() > 5: # Đảm bảo có đủ pixel để khởi tạo
            
            # --- PHƯƠNG PHÁP 1 (Hiện tại): Mean của các vùng tin cậy cao ---
            # Lấy trung bình của tất cả các feature vector trong vùng có CAM > ngưỡng.
            # Ưu điểm: Ổn định, ít bị ảnh hưởng bởi nhiễu.
            # Nhược điểm: Có thể bị "pha loãng" nếu vùng tin cậy bao gồm nhiều dạng
            #            hình thái khác nhau.
            # features_for_class = features_reshaped[high_conf_mask.flatten()]
            # centroid = torch.mean(features_for_class, dim=0)

            # # --- PHƯƠNG PHÁP 2 (Tiềm năng): Lấy Feature của Pixel có CAM cao nhất ---
            # # Chỉ lấy feature vector của pixel có điểm CAM cao nhất làm tâm.
            # # Ưu điểm: Đại diện cho vùng "đặc trưng nhất", "tinh túy nhất" của lớp.
            # # Nhược điểm: Rất nhạy với nhiễu, nếu pixel cao nhất là một điểm bất thường
            # #            thì tâm khởi tạo sẽ rất tệ.
            # peak_pixel_index = torch.argmax(class_cam_scores)
            # centroid = features_reshaped[peak_pixel_index]
            
            # # --- PHƯƠNG PHÁP 3 (Tiềm năng): Weighted Mean dùng CAM làm trọng số ---
            # # Lấy trung bình có trọng số, các pixel có CAM cao hơn sẽ đóng góp nhiều hơn
            # # vào tâm cụm. Đây là phiên bản nâng cấp của Phương pháp 1.
            # # Ưu điểm: Vẫn ổn định nhưng tâm sẽ lệch về phía các vùng đặc trưng hơn,
            # #          có khả năng cho kết quả tốt hơn.
            # # Nhược điểm: Tính toán phức tạp hơn một chút.
            features_for_class = features_reshaped[high_conf_mask.flatten()]
            device = features_reshaped.device

            features_for_class = features_for_class.to(device)
            weights = class_cam_scores[high_conf_mask]
            weights = weights.to(device)

            centroid = torch.sum(weights.view(-1, 1) * features_for_class, dim=0) / torch.sum(weights)

            # # --- PHƯƠNG PHÁP 4 (Tiềm năng): Lấy mẫu ngẫu nhiên từ vùng tin cậy cao ---
            # # Chọn ngẫu nhiên một pixel từ các vùng có CAM > ngưỡng.
            # # Ưu điểm: Đơn giản, có thể giúp thoát khỏi các điểm tối ưu cục bộ (ít relevant
            # #          hơn ở đây vì chúng ta chỉ chạy 1 lần khởi tạo).
            # # Nhược điểm: Kết quả không có tính tất định (non-deterministic), mỗi lần chạy
            # #             có thể ra kết quả khác nhau trừ khi cố định random seed.
            # features_for_class = features_reshaped[high_conf_mask.flatten()]
            # random_index = torch.randint(0, len(features_for_class), (1,)).item()
            # centroid = features_for_class[random_index]
            
            initial_centroids.append(centroid)
            valid_present_classes.append(class_idx)

    # ==================================================================
    #    PHẦN CÒN LẠI CỦA THUẬT TOÁN (KHÔNG THAY ĐỔI)
    # ==================================================================
    k = len(initial_centroids)
    if k == 0: return np.full(orig_img_shape, 4, dtype=np.int32)
    
    centroids = torch.stack(initial_centroids).to(device)
    features_reshaped = features_reshaped.to(device)
    
    for i in range(n_iters):
        similarity_matrix = cosine_similarity_pytorch(features_reshaped, centroids)
        cluster_assignments = torch.argmax(similarity_matrix, dim=1)
        new_centroids = []
        for j in range(k):
            members = features_reshaped[cluster_assignments == j]
            if len(members) > 0: new_center = members.mean(dim=0)
            else: new_center = centroids[j]
            new_centroids.append(new_center)
        centroids = torch.stack(new_centroids)
        
    final_similarity = cosine_similarity_pytorch(features_reshaped, centroids).cpu()
    score_map_low_res = final_similarity.reshape(H, W, k).permute(2, 0, 1)
    score_map_tensor = score_map_low_res.float().unsqueeze(0)
    upsampled_scores = F.interpolate(score_map_tensor, size=orig_img_shape, mode='bilinear', align_corners=False).squeeze(0).numpy()
    class_mapping = np.array(valid_present_classes)
    final_mask = class_mapping[np.argmax(upsampled_scores, axis=0)]
    
    return final_mask
# =======================================================================
#    HÀM INFERENCE GỐC ĐƯỢC NÂNG CẤP VỚI "CÔNG TẮC"
# =======================================================================

def infer_baseline(model,iters, dataroot, n_class, args):
    """
    Thực hiện inference. Logic sẽ thay đổi dựa trên `args.method`.
    Đã sửa lại logic xử lý background để khớp với iouutils.py gốc.
    """
    method = getattr(args, 'method', 'cam')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = BaselineInferDataset(dataroot=dataroot, n_class=n_class, transform=transform)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, batch_size=1)
    
    pred_list = []
    gt_list = []

    for img_name, img_tensor, label_tensor in tqdm(infer_data_loader, desc=f"Inferring with [{method.upper()}]"):
        img_name = img_name[0]
        image_label = label_tensor[0]
        img_tensor = img_tensor.to(device)
        
        orig_img_path = os.path.join(dataroot, 'img/', img_name + '.png')
        orig_img_pil = Image.open(orig_img_path).convert("RGB")
        orig_img_shape = np.array(orig_img_pil).shape[:2]

        # Lấy ground truth trước
        gt_map_path = os.path.join(dataroot, 'mask/', img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)

        # Tạo mask dự đoán
        if method == 'cosine':
            pred_mask = refine_with_cosine_kmeans(model, img_tensor, image_label, orig_img_shape,n_iters=iters)
        else: # Mặc định là 'cam'
            with torch.no_grad():
                cam_raw = model(img_tensor)
            cam_upsampled = F.interpolate(cam_raw, size=orig_img_shape, mode='bilinear', align_corners=False)[0]
            cam_scores = cam_upsampled.cpu().numpy() * image_label.clone().view(n_class, 1, 1).numpy()
            pred_mask = np.argmax(cam_scores, axis=0)

        # ***** DÒNG SỬA LỖI QUAN TRỌNG NHẤT *****
        # Áp dụng logic từ iouutils.py: ép các pixel là background trong GT
        # cũng phải là background trong dự đoán.
        pred_mask[gt_map == 4] = 4
        
        pred_list.append(pred_mask)
        
    return gt_list, pred_list