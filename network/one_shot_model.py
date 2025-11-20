# network/one_shot_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from network import resnet38d_baseline # Import backbone baseline của chúng ta

class OneShotModel(nn.Module):
    def __init__(self, n_class, pretrained_path, freeze_encoder=True, fine_tune_layer='b7'):
        """
        Model cho One-Shot Segmentation.
        
        Args:
            n_class (int): Số lớp foreground.
            pretrained_path (str): Đường dẫn đến checkpoint của Giai đoạn 1.
            freeze_encoder (bool): Nếu True, đóng băng phần lớn Encoder.
            fine_tune_layer (str): Tên của layer cuối cùng sẽ được fine-tune (ví dụ: 'b7').
                                   Tất cả các layer trước đó sẽ bị đóng băng.
        """
        super().__init__()
        self.n_class = n_class
        
        # 1. Khởi tạo Encoder
        self.encoder = resnet38d_baseline.Net()
        print("Đang khởi tạo Encoder...")
        
        # 2. Load trọng số đã được huấn luyện trước từ Giai đoạn 1
        print(f"Đang tải trọng số Giai đoạn 1 từ: {pretrained_path}")
        # Checkpoint của Giai đoạn 1 chứa trọng số của cả backbone và classifier heads
        # Chúng ta chỉ cần load trọng số của backbone
        full_state_dict = torch.load(pretrained_path)
        encoder_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('backbone.'):
                # Loại bỏ tiền tố 'backbone.' để khớp với self.encoder
                new_key = key.replace('backbone.', '', 1)
                encoder_state_dict[new_key] = value
        
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        print("Tải trọng số Encoder thành công.")

        # 3. Đóng băng các layer của Encoder
        if freeze_encoder:
            self.freeze_layers(fine_tune_layer)
        
    def freeze_layers(self, fine_tune_layer):
        """
        Đóng băng tất cả các tham số của encoder ngoại trừ layer được chỉ định.
        """
        print(f"Đóng băng các layer của Encoder, chỉ fine-tune từ layer '{fine_tune_layer}' trở đi.")
        
        # Tìm xem đã đến layer cần fine-tune chưa
        fine_tuning_started = False
        total_params = 0
        unfrozen_params = 0
        
        for name, param in self.encoder.named_parameters():
            total_params += 1
            if fine_tune_layer in name:
                fine_tuning_started = True
            
            if fine_tuning_started:
                param.requires_grad = True # Mở băng layer này và các layer sau nó
                unfrozen_params += 1
            else:
                param.requires_grad = False # Đóng băng

        print(f"Hoàn tất đóng băng. {unfrozen_params}/{total_params} tham số trong Encoder sẽ được cập nhật.")

    def forward(self, support_img, support_mask, query_img):
        """
        Thực hiện quá trình forward cho one-shot segmentation.
        
        Args:
            support_img (Tensor): Ảnh support [B, C, H, W]
            support_mask (Tensor): Mask của ảnh support [B, H, W]
            query_img (Tensor): Ảnh query [B, C, H, W]
            
        Returns:
            Tensor: Logits dự đoán cho ảnh query [B, n_class, H_feat, W_feat]
        """
        # --- Encoder ---
        # Cho cả support và query đi qua encoder
        # Chúng ta chỉ cần feature map cuối cùng (conv6) có shape [B, 4096, H_feat, W_feat]
        _, _, support_feat = self.encoder(support_img)
        _, _, query_feat = self.encoder(query_img)
        
        # --- Tạo Prototypes từ Support Feature Map ---
        prototypes = self._create_prototypes(support_feat, support_mask)
        
        # --- Phân đoạn Query Feature Map ---
        # Tính toán độ tương đồng cosine và trả về logits
        query_logits = self._calculate_cosine_similarity(query_feat, prototypes)
        
        return query_logits

    def _create_prototypes(self, support_feat, support_mask):
        """
        Tạo các vector prototype cho từng lớp từ support feature map.
        ĐÃ SỬA LẠI ĐỂ XỬ LÝ TỪNG ẢNH TRONG BATCH.
        """
        batch_size, feat_channels, feat_h, feat_w = support_feat.shape
        
        # Resize support_mask về cùng kích thước với feature map
        support_mask_resized = F.interpolate(support_mask.unsqueeze(1).float(), 
                                             size=(feat_h, feat_w), 
                                             mode='nearest').squeeze(1).long()

        # Chuẩn bị một tensor rỗng để chứa tất cả các prototype
        all_prototypes = torch.zeros((batch_size, feat_channels, self.n_class), device=support_feat.device)

        # Lặp qua từng ảnh trong batch
        for i in range(batch_size):
            # Lấy feature map và mask của ảnh thứ i
            feat_i = support_feat[i] # Shape: [C, H_feat, W_feat]
            mask_i = support_mask_resized[i] # Shape: [H_feat, W_feat]
            
            for c in range(self.n_class): # Lặp qua từng lớp foreground
                # Tạo mask nhị phân cho lớp c trên ảnh i
                class_mask = (mask_i == c)
                
                # Nếu không có pixel nào của lớp này, prototype sẽ là vector 0 (mặc định)
                if class_mask.sum() == 0:
                    continue # Bỏ qua và giữ prototype là 0
                
                # Lấy tất cả các feature vector thuộc lớp này
                # feat_i.transpose(0, 1) -> [H, C, W]
                # feat_i.permute(1, 2, 0) -> [H, W, C]
                # class_features sẽ có shape [num_pixels, C]
                class_features = feat_i.permute(1, 2, 0)[class_mask]
                
                # Tính prototype bằng cách lấy trung bình
                # prototype_ic có shape [C]
                prototype_ic = class_features.mean(dim=0)
                
                # Gán prototype đã tính vào tensor tổng
                all_prototypes[i, :, c] = prototype_ic
        
        return all_prototypes

    def _calculate_cosine_similarity(self, query_feat, prototypes):
        """
        Tính toán độ tương đồng cosine giữa mỗi pixel của query và các prototypes.
        """
        batch_size, feat_channels, feat_h, feat_w = query_feat.shape
        
        # Chuẩn hóa L2 cho các feature vector của query
        query_feat_normalized = F.normalize(query_feat, p=2, dim=1)
        
        # Chuẩn hóa L2 cho các prototype
        prototypes_normalized = F.normalize(prototypes, p=2, dim=1)
        
        # Reshape query feature map để thực hiện phép nhân ma trận
        query_reshaped = query_feat_normalized.view(batch_size, feat_channels, -1) # [B, C, H*W]
        
        # Tính độ tương đồng cosine
        # (B, C, H*W).T * (B, C, n_class) -> (B, H*W, n_class)
        similarity = torch.bmm(query_reshaped.transpose(1, 2), prototypes_normalized)
        
        # Reshape trở lại dạng feature map
        # (B, H*W, n_class) -> (B, n_class, H*W) -> (B, n_class, H, W)
        logits = similarity.transpose(1, 2).view(batch_size, self.n_class, feat_h, feat_w)
        
        # (Tùy chọn) Nhân với một hệ số scale để logits có giá trị lớn hơn, giúp hội tụ tốt hơn
        logits = logits * 10.0
        
        return logits