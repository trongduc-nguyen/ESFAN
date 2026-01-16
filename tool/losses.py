import torch
import torch.nn as nn
import torch.nn.functional as F

class JointTrainingLosses(nn.Module):
    def __init__(self, n_tissue_class=4, contrastive_temp=0.07, class_weights=None):
        super().__init__()
        self.n_tissue = n_tissue_class
        # [SỬA] Bỏ self.other_idx
        self.temp = contrastive_temp
        
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.ones(n_tissue_class)

    def patch_classification_loss(self, patch_logits, img_labels, thresholds):
        """
        Logic 4-Class:
        - Mọi patch đều dự đoán ra 1 trong 4 class.
        - Lọc 'Nền' bằng cách xem Confidence có vượt qua Threshold hay không.
        """
        device = patch_logits.device
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)

        B_total = patch_logits.size(0)
        B_img = img_labels.size(0)
        patches_per_img = B_total // B_img
        
        # 1. Lấy xác suất
        probs = F.softmax(patch_logits, dim=1)      # [N, 4]
        max_probs, preds = torch.max(probs, dim=1)  # [N], [N]
        
        img_labels_expanded = img_labels.unsqueeze(1).expand(-1, patches_per_img, -1).reshape(B_total, -1)
        
        # [SỬA] KHÔNG LỌC patch 'Other' nữa vì không có class đó.
        # Chúng ta dùng toàn bộ patch để xét duyệt
        
        # Kiểm tra class dự đoán có trong nhãn ảnh không
        pred_class_exists = img_labels_expanded.gather(1, preds.unsqueeze(1)).squeeze(1)

        loss = torch.tensor(0.0, device=device)
        count = 0
        
        # Stats
        pos_count = 0
        neg_count = 0

        # CASE 1: NEGATIVE LEARNING (Class dự đoán KHÔNG có trong ảnh)
        # Patch dự đoán Tumor, nhưng ảnh không có Tumor -> Phạt
        neg_mask = (pred_class_exists == 0)
        
        if neg_mask.sum() > 0:
            neg_probs = probs[neg_mask, preds[neg_mask]]
            neg_cls_idx = preds[neg_mask]
            curr_neg_weights = self.class_weights[neg_cls_idx]
            
            # Minimize prob -> Đẩy xác suất này xuống thấp
            neg_loss = -(curr_neg_weights * torch.log(1 - neg_probs + 1e-8)).sum()
            
            loss += neg_loss
            count += neg_mask.sum()
            neg_count = neg_mask.sum().item()

        # CASE 2: POSITIVE LEARNING (Class dự đoán CÓ trong ảnh)
        pos_candidate_mask = (pred_class_exists == 1)
        
        if pos_candidate_mask.sum() > 0:
            # Lấy ngưỡng
            if isinstance(thresholds, torch.Tensor):
                target_threshs = thresholds[preds]
            else:
                target_threshs = thresholds
            
            # [QUAN TRỌNG] Chỉ học những patch có độ tin cậy CAO.
            # Những patch tin cậy thấp (dù đúng class trong ảnh) có thể là Nền -> Bỏ qua.
            conf_mask = (max_probs > target_threshs)
            
            final_pos_mask = pos_candidate_mask & conf_mask
            
            if final_pos_mask.sum() > 0:
                pos_probs = probs[final_pos_mask, preds[final_pos_mask]]
                pos_cls_idx = preds[final_pos_mask]
                curr_pos_weights = self.class_weights[pos_cls_idx]
                
                pos_loss = -(curr_pos_weights * torch.log(pos_probs + 1e-8)).sum()
                
                loss += pos_loss
                count += final_pos_mask.sum()
                pos_count = final_pos_mask.sum().item()
        
        if count > 0:
            loss = loss / count
            
        stats = {
            "pos_cnt": pos_count, 
            "neg_cnt": neg_count,
            "total_patch": B_total
        }
        return loss, stats

    def patch_contrastive_loss(self, embeddings, patch_logits):
        """
        Contrastive Loss cho 4 Class.
        Chỉ dùng những patch có confidence cao để tránh kéo patch nền vào cluster mô.
        """
        # [SỬA] Thêm logic lọc theo confidence để loại bỏ nhiễu nền
        probs = F.softmax(patch_logits, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        
        # Ngưỡng cứng cho Contrastive (ví dụ 0.8) để đảm bảo chỉ cluster những thằng xịn
        # Hoặc dùng threshold động từ bên ngoài truyền vào (nhưng để đơn giản ta dùng fix)
        valid_mask = (max_probs > 0.1) 
        
        if valid_mask.sum() < 2: 
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        feats = embeddings[valid_mask]
        labels = pseudo_labels[valid_mask]
        
        # ... (Đoạn tính Sim Matrix và Loss giữ nguyên như cũ) ...
        sim_matrix = torch.matmul(feats, feats.T) / self.temp
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        diag_mask = torch.eye(feats.size(0), device=embeddings.device)
        pos_mask = pos_mask - diag_mask
        
        if pos_mask.sum() == 0: return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        exp_sim = torch.exp(sim_matrix)
        denom = exp_sim.sum(dim=1, keepdim=True) - torch.exp(torch.tensor(1.0/self.temp, device=embeddings.device))
        log_prob = sim_matrix - torch.log(denom + 1e-8)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        
        loss = - mean_log_prob_pos.mean()
        
        return loss