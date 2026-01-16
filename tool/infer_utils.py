import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ==============================================================================
# MODULE: Multi-Prototype Weighted Similarity (User's Idea - Adapted)
# ==============================================================================
class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw):
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.kernel_size = [int(ft_l // grid_l) for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(self.kernel_size)

    def safe_norm(self, x, p=2, dim=1, eps=1e-4):
        x_norm = torch.norm(x, p=p, dim=dim)
        x_norm = torch.max(x_norm, torch.ones_like(x_norm, device=x.device) * eps)
        x = x.div(x_norm.unsqueeze(1).expand_as(x))
        return x
        
    def forward(self, qry_fts, sup_fts, sup_mask, true_bg_mask=None, mode='gridconv+', thresh=0.95):
        batch_size, n_channels, h_fts, w_fts = qry_fts.shape
        
        # Cập nhật kernel size động
        curr_kernel = [int(h_fts // self.proto_grid[0]), int(w_fts // self.proto_grid[1])]
        curr_kernel = [max(1, k) for k in curr_kernel]
        
        if curr_kernel != self.kernel_size:
            avg_pool_func = lambda x: F.avg_pool2d(x, curr_kernel)
        else:
            avg_pool_func = self.avg_pool_op

        if mode == 'mask':  
            # Global only code (giữ nguyên)
            numerator = torch.sum(sup_fts * sup_mask, dim=(-1, -2))
            denominator = sup_mask.sum(dim=(-1, -2)) + 1e-5
            proto = numerator / denominator 
            pred_mask = F.cosine_similarity(qry_fts, proto.unsqueeze(-1).unsqueeze(-1), dim=1, eps=1e-4)
            return pred_mask.unsqueeze(1)

        elif mode == 'gridconv+':
            # 1. Local Prototypes
            local_sup_fts = avg_pool_func(sup_fts)
            local_sup_mask = avg_pool_func(sup_mask)
            if true_bg_mask is not None:
                local_true_bg_mask = avg_pool_func(true_bg_mask)
            
            local_sup_fts = local_sup_fts.view(batch_size, n_channels, -1).permute(0, 2, 1)
            local_sup_mask = local_sup_mask.view(batch_size, 1, -1).permute(0, 2, 1)
            if true_bg_mask is not None:
                local_true_bg_mask = local_true_bg_mask.view(batch_size, 1, -1).permute(0, 2, 1)

            # 2. Global Prototype
            global_proto = torch.sum(sup_fts * sup_mask, dim=(-1, -2)) / (sup_mask.sum(dim=(-1, -2)) + 1e-5)
            
            scores_list = []
            for i in range(batch_size):
                valid_signal_mask = local_sup_mask[i].squeeze() > thresh
                
                if true_bg_mask is not None:
                    not_true_bg_mask = local_true_bg_mask[i].squeeze() < 0.5
                    final_valid_mask = valid_signal_mask & not_true_bg_mask
                else:
                    final_valid_mask = valid_signal_mask
                
                valid_local_protos = local_sup_fts[i][final_valid_mask]
                all_protos = torch.cat([valid_local_protos, global_proto[i:i+1]], dim=0) # [N_proto, C]

                if len(all_protos) == 0:
                    scores_list.append(torch.zeros_like(qry_fts[i:i+1, 0:1, ...]))
                    continue
                    
                # --- SỬA LỖI TẠI ĐÂY ---
                # all_protos: [N_proto, C]
                # Ta normalize theo chiều C (dim=1)
                protos_norm = self.safe_norm(all_protos, dim=1) # [N_proto, C]
                
                # Reshape thành Weight cho Conv2d: [Out(N_proto), In(C), 1, 1]
                weight = protos_norm.view(-1, n_channels, 1, 1)
                
                qry_norm = self.safe_norm(qry_fts[i:i+1]) # [1, C, H, W]
                
                # Thực hiện Conv2d
                dists = F.conv2d(qry_norm, weight) # [1, N_proto, H, W]
                
                # Weighted Sum (Attention)
                attention = F.softmax(dists, dim=1) 
                pred_grid = torch.sum(attention * dists, dim=1, keepdim=True) 
                
                scores_list.append(pred_grid)
                
            return torch.cat(scores_list, dim=0)
        else:
            return None
# ==============================================================================
# EXISTING UTILS
# ==============================================================================

def normalize_cam(cam):
    """Chuẩn hóa CAM về khoảng [0, 1] cho từng lớp trong từng ảnh."""
    B, C, H, W = cam.shape
    cam_min = cam.view(B, C, -1).min(dim=2)[0].view(B, C, 1, 1)
    cam_max = cam.view(B, C, -1).max(dim=2)[0].view(B, C, 1, 1)
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return norm_cam
def tri_scale_gated_fusion(cam_dict, target_size):
    """
    Module 1: Auto-Adaptive Tri-Scale Fusion
    Thay thế các tham số cứng (0.7, 0.5) bằng trọng số động dựa trên độ đo sự nhất quán (Consistency).
    """
    cam1 = cam_dict['cam1'] 
    cam2 = cam_dict['cam2'] 
    cam8 = cam_dict['cam8'] 

    # 1. Upsample về cùng kích thước
    cam1_up = F.interpolate(cam1, size=target_size, mode='bilinear', align_corners=False)
    cam2_up = F.interpolate(cam2, size=target_size, mode='bilinear', align_corners=False)
    cam8_up = F.interpolate(cam8, size=target_size, mode='bilinear', align_corners=False)

    # 2. Chuẩn hóa về [0, 1] để so sánh công bằng
    cam1_norm = normalize_cam(cam1_up)
    cam2_norm = normalize_cam(cam2_up)
    cam8_norm = normalize_cam(cam8_up) # Dùng bản norm để tính toán trọng số
    
    # Gate từ High-level (Semantic Anchor)
    gate = torch.sigmoid(cam8_up)

    # --- 3. TÍNH TOÁN TRỌNG SỐ TỰ ĐỘNG (SMART PART) ---
    # Tư duy: Tính độ tương đồng (Cosine Similarity hoặc Overlap) giữa Low-level và High-level
    # Chỉ tính trên vùng Foreground (nơi Gate > 0.5) để loại bỏ background noise
    
    # Tạo mask vùng quan tâm (ROI) từ cam8
    roi_mask = (gate > 0.1).float() # Lấy vùng có tín hiệu semantic (dù yếu)
    
    # Tính độ khớp (Consistency Score) cho từng Class trong từng Batch
    # Score = Sum(Low * High * ROI) / Sum(High * ROI)
    # Ý nghĩa: Low-level feature phủ được bao nhiêu % năng lượng của High-level feature?
    
    def get_consistency_score(low, high, mask):
        # low, high: [B, C, H, W]
        intersection = (low * high * mask).sum(dim=(2, 3)) 
        union = (high * mask).sum(dim=(2, 3)) + 1e-6
        score = intersection / union
        # Score ra [B, C]. Ta lấy trung bình các class để ra 1 số alpha chung cho ảnh, 
        # hoặc giữ nguyên để alpha riêng cho từng class (tốt hơn).
        return score.mean(dim=1).view(-1, 1, 1, 1) # [B, 1, 1, 1]

    # Alpha (cho cam1 - low) và Beta (cho cam2 - mid)
    # Nhân thêm hệ số scale (ví dụ 1.5) để boost tín hiệu nếu khớp tốt
    # Clip để không quá lớn hoặc quá nhỏ
    
    alpha_dynamic = get_consistency_score(cam1_norm, cam8_norm, roi_mask) * 1.5
    alpha_dynamic = torch.clamp(alpha_dynamic, 0.1, 0.8) # Giới hạn [0.1, 0.8]
    
    beta_dynamic = get_consistency_score(cam2_norm, cam8_norm, roi_mask) * 1.5
    beta_dynamic = torch.clamp(beta_dynamic, 0.3, 1.0)   # Mid-level thường tin cậy hơn Low-level

    # --- 4. FUSION ---
    # Công thức cũ: cam8 + 0.7 * ... + 0.5 * ...
    # Công thức mới: Tự động theo từng ảnh
    
    cam_fused = cam8_up + beta_dynamic * (cam2_norm * gate) + alpha_dynamic * (cam1_norm * gate)
    
    return cam_fused

def prototype_refinement(cam_fused, features, img_label, target_size, n_class, conf_thresh=0.85):
    """
    Module 2: Feature-Prototype Rectification (PR) - UPGRADED with MultiProtoAsConv
    Using weighted similarity of Local + Global prototypes.
    """
    B, C, Hf, Wf = features.shape
    
    # 1. Chuẩn bị module MultiProto
    # Grid size: 8x8 là kích thước tốt để bắt texture local trên feature map 32x32
    proto_layer = MultiProtoAsConv(proto_grid=[8, 8], feature_hw=[Hf, Wf]).to(features.device)
    
    # Chuẩn bị background mask (Low-res) để lọc prototype
    # Chúng ta cần resize ảnh gốc hoặc resize mask high-res về size feature
    # Ở đây để đơn giản và nhanh, ta không tạo true_bg_mask từ ảnh gốc trong hàm này 
    # (vì tốn chi phí đọc lại ảnh). Ta sẽ lọc dựa trên ngưỡng confidence thấp của chính CAM.
    # Tuy nhiên, nếu muốn dùng 'true_bg_mask' như logic bạn đề xuất, ta cần mask background.
    # Ở đây tôi dùng '1 - Foreground_CAM' làm proxy cho bg_mask trong feature space.
    
    cam_prob = normalize_cam(cam_fused) # [B, N_class, H_orig, W_orig]
    
    # Downsample CAM về size feature để làm support mask
    cam_prob_low = F.interpolate(cam_prob, size=(Hf, Wf), mode='bilinear', align_corners=False)
    
    refined_cam_list = []
    
    # 2. Xử lý từng lớp
    for c in range(n_class):
        # Lấy mask của lớp hiện tại
        # [B, 1, Hf, Wf]
        sup_mask = cam_prob_low[:, c:c+1] 
        
        # Kiểm tra xem ảnh có chứa lớp này không (từ nhãn ảnh)
        # Nếu không có, similarity = 0
        if img_label[:, c].sum() == 0:
            refined_cam_list.append(torch.zeros_like(cam_fused[:, c:c+1]))
            continue
            
        # Proxy cho True Background: Những vùng mà tổng xác suất foreground rất thấp
        # Hoặc đơn giản là 1 - sup_mask nếu coi binary.
        # Để an toàn, ta pass true_bg_mask=None và chỉ dựa vào threshold của sup_mask > 0.95
        # Nếu bạn muốn truyền bg_mask thực sự, cần sửa pipeline để truyền từ ngoài vào.
        # Ở mức độ này, lọc bằng sup_mask > thresh là đủ mạnh.
        
        # Tính Similarity Map bằng MultiProtoAsConv (Weighted)
        # Input features là low-res (conv6), output sim_map cũng là low-res
        sim_map_low = proto_layer(features, features, sup_mask, mode='gridconv+', thresh=conf_thresh)
        
        # Upsample Similarity Map về kích thước gốc
        sim_map_up = F.interpolate(sim_map_low, size=target_size, mode='bilinear', align_corners=False)
        
        # Rectification: Mix giữa CAM gốc và Similarity Map
        # Best Params tuning: lambda=0.3
        lambda_param = 0.3
        
        # Chỉ refine nếu có prototype (tức là sim_map có giá trị)
        # Nếu sim_map toàn 0 (do không tìm thấy proto), giữ nguyên CAM
        has_proto_mask = (sim_map_up.abs().sum(dim=(2,3), keepdim=True) > 1e-6).float()
        
        cam_c_refined = (1 - lambda_param) * cam_fused[:, c:c+1] + lambda_param * sim_map_up
        cam_c_refined = (1 - has_proto_mask) * cam_fused[:, c:c+1] + has_proto_mask * cam_c_refined
        
        refined_cam_list.append(cam_c_refined)
        
    refined_cam = torch.cat(refined_cam_list, dim=1)
    
    return refined_cam

def gen_bg_mask(img_np):
    """Giữ nguyên logic tạo background mask"""
    import cv2
    from skimage import morphology
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    binary_mask = (binary == 255)
    bg_mask = morphology.remove_small_objects(binary_mask, min_size=50, connectivity=1)
    
    return bg_mask.astype(np.float32)