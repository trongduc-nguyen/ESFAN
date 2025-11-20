# network/multitask_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from network import resnet38d_baseline

class MultiTaskNet(nn.Module):
    def __init__(self, n_class, proto_grid_size=(14, 14)):
        """
        Model Multi-Task học đồng thời Classification và Segmentation Consistency.
        
        Args:
            n_class (int): Số lớp foreground.
            proto_grid_size (tuple): Kích thước lưới cho local prototypes.
        """
        super(MultiTaskNet, self).__init__()
        self.n_class = n_class
        self.proto_grid_size = proto_grid_size

        # 1. Khởi tạo Encoder (Backbone)
        self.backbone = resnet38d_baseline.Net()

        # 2. Khởi tạo các đầu phân loại (Classifier Heads)
        # Giữ nguyên như phiên bản baseline
        self.dropout7 = nn.Dropout2d(0.5)
        self.ic1 = nn.Conv2d(512, n_class, 1)
        self.ic2 = nn.Conv2d(1024, n_class, 1)
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        self.from_scratch_layers = [self.ic1, self.ic2, self.fc8]
        self._init_weights()

    def _init_weights(self):
        for m in self.from_scratch_layers:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, x_query=None, pseudo_mask_support=None):
        """
        Hàm forward linh hoạt.
        
        - Nếu chỉ có `x`: Chạy chế độ Classification, trả về logits và CAM.
        - Nếu có cả `x_query` và `pseudo_mask_support`: Chạy thêm one-shot matching.
        """
        # --- Nhiệm vụ 1: Classification (luôn được thực hiện) ---
        
        # Lấy feature maps từ backbone
        b_45, b_52, conv6 = self.backbone(x)
        
        # Tính logits phân loại (cho L_cls)
        # Logic này giống hệt resnet38_cls_baseline.py
        pooled_b45 = F.adaptive_avg_pool2d(b_45, (1, 1))
        pooled_b52 = F.adaptive_avg_pool2d(b_52, (1, 1))
        
        x1 = self.ic1(pooled_b45).squeeze(-1).squeeze(-1)
        x2 = self.ic2(pooled_b52).squeeze(-1).squeeze(-1)
        
        pooled_conv6 = F.adaptive_avg_pool2d(self.dropout7(conv6), (1, 1))
        x3 = self.fc8(pooled_conv6).squeeze(-1).squeeze(-1)
        
        # Tạo CAM (sẽ được dùng để tạo nhãn giả "tức thời")
        cam = F.conv2d(conv6, self.fc8.weight)
        cam = F.relu(cam) # CAM thô
        
        # Nếu không có query, chỉ trả về kết quả classification
        if x_query is None:
            return x1, x2, x3, cam

        # --- Nhiệm vụ 2: One-Shot Matching (nếu có đầu vào) ---
        
        # `x` bây giờ đóng vai trò support
        support_feat = conv6
        
        # Lấy feature map của query
        _, _, query_feat = self.backbone(x_query)
        
        # Lặp qua từng lớp để tính one-shot segmentation logits
        all_class_logits = []
        feat_h, feat_w = support_feat.shape[-2:]
            
        # `pseudo_mask_support` có shape [B, H, W]
        # Chuyển nó về kích thước của feature map
        support_mask_resized = F.interpolate(pseudo_mask_support.unsqueeze(1).float(), 
                                             size=(feat_h, feat_w), 
                                             mode='nearest').squeeze(1).long()
        
        for c in range(self.n_class):
            class_mask_binary = (support_mask_resized == c).float().unsqueeze(1)
            
            if class_mask_binary.sum() > 0:
                prototypes_c = self._create_multi_prototypes_for_class(support_feat, class_mask_binary)
            else:
                prototypes_c = torch.zeros((1, support_feat.shape[1]), device=support_feat.device)

            class_logits = self._match_with_prototypes(query_feat, prototypes_c)
            all_class_logits.append(class_logits)
            
        seg_logits = torch.cat(all_class_logits, dim=1)
        
        return x1, x2, x3, cam, seg_logits

    # --- Các hàm One-Shot (sao chép từ one_shot_model.py) ---

    def _create_multi_prototypes_for_class(self, support_feat, class_mask_binary):
        # ... (Sao chép y hệt code từ one_shot_model.py)
        # 1. Tạo Prototype Toàn cục (Global Prototype)
        fg_sum = torch.sum(support_feat * class_mask_binary, dim=(2, 3))
        fg_area = torch.sum(class_mask_binary, dim=(2, 3))
        global_proto = fg_sum / (fg_area + 1e-6) # Shape: [B, C]
        
        # 2. Tạo Prototype Địa phương (Local Prototypes)
        feat_h, feat_w = support_feat.shape[-2:]
        grid_h, grid_w = self.proto_grid_size
        pool_kernel_size = (feat_h // grid_h, feat_w // grid_w)
        
        local_proto_candidates = F.avg_pool2d(support_feat, kernel_size=pool_kernel_size)
        local_mask = F.avg_pool2d(class_mask_binary, kernel_size=pool_kernel_size)
        
        local_proto_candidates = local_proto_candidates.view(support_feat.shape[0], support_feat.shape[1], -1).permute(0, 2, 1)
        local_mask = local_mask.view(support_feat.shape[0], -1)
        
        local_prototypes = []
        for i in range(support_feat.shape[0]):
            protos_i = local_proto_candidates[i][local_mask[i] > 0.5]
            if protos_i.shape[0] > 0:
                local_prototypes.append(protos_i)
        
        final_prototypes = global_proto
        if len(local_prototypes) > 0:
            final_prototypes = torch.cat([global_proto, local_prototypes[0]], dim=0)
            
        return final_prototypes

    def _match_with_prototypes(self, query_feat, prototypes):
        # ... (Sao chép y hệt code từ one_shot_model.py, phiên bản có weighted sum)
        TEMPERATURE = 20.0
        query_normalized = F.normalize(query_feat, p=2, dim=1)
        prototypes_normalized = F.normalize(prototypes, p=2, dim=1)
        
        proto_as_kernel = prototypes_normalized.unsqueeze(-1).unsqueeze(-1)
        
        similarities = F.conv2d(query_normalized, proto_as_kernel) * TEMPERATURE
        
        attention_weights = F.softmax(similarities, dim=1)
        class_logits = torch.sum(attention_weights * similarities, dim=1, keepdim=True)
        
        return class_logits
        
    def get_parameter_groups(self):
        # Giữ nguyên hàm này từ resnet38_cls_baseline.py
        # Để có thể set các learning rate khác nhau
        groups = ([], [], [], []) 
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            
            is_from_scratch = any(id(p) == id(layer_p) for layer in self.from_scratch_layers for layer_p in layer.parameters())

            if 'backbone' in name:
                if 'b2.' in name or 'b3.' in name or 'b4.' in name:
                    groups[0].append(p)
                else:
                    groups[1].append(p)
            else: # classification heads
                if 'weight' in name:
                    groups[2].append(p)
                else:
                    groups[3].append(p)
        return groups