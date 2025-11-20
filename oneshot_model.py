# File: oneshot_model.py (CẬP NHẬT ĐỂ TRUYỀN THAM SỐ TRUE_BG_MASK)

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.resnet38d_baseline import Net as ResNet38Backbone
from tool.alpmodule import MultiProtoAsConv
from network import resnet38d
import os

class OneShotSegModel(nn.Module):
    def __init__(self, pretrained_path=None):
        """
        Khởi tạo model.
        Args:
            pretrained_path (str, optional): Đường dẫn đến file checkpoint.
                - Nếu là file .params: Load trọng số ImageNet cho encoder.
                - Nếu là file .pth:
                    + Nếu key có 'encoder.': Load toàn bộ state_dict của OneShotSegModel.
                    + Nếu không: Load state_dict chỉ cho encoder.
        """
        super().__init__()
        
        # --- 1. Khởi tạo các thành phần của model ---
        self.encoder = ResNet38Backbone()
        self.classifier = MultiProtoAsConv(proto_grid=(14, 14), feature_hw=(28, 28))
        self.logit_scale = nn.Parameter(torch.ones([]) * 20.0)
        
        # --- 2. LOGIC LOAD TRỌNG SỐ NÂNG CẤP ---
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Đang tải trọng số từ: {pretrained_path}")
            
            if pretrained_path.endswith('.params'):
                # Trường hợp 1: Load từ file .params của MXNet (chỉ cho encoder)
                print("Phát hiện file .params, đang chuyển đổi và load cho encoder...")
                weights_dict = resnet38d.convert_mxnet_to_torch(pretrained_path)
                self.encoder.load_state_dict(weights_dict, strict=False)
                print("Tải trọng số encoder từ file .params thành công.")

            elif pretrained_path.endswith('.pth'):
                # Trường hợp 2: Load từ file .pth của PyTorch
                state_dict = torch.load(pretrained_path, map_location='cpu')
                
                if any(key.startswith('encoder.') for key in state_dict.keys()):
                    print("Phát hiện checkpoint của OneShotSegModel, đang load toàn bộ model...")
                    self.load_state_dict(state_dict)
                    print("Load toàn bộ model thành công.")
                else:
                    print("Phát hiện checkpoint chỉ chứa backbone, đang load cho encoder...")
                    self.encoder.load_state_dict(state_dict)
                    print("Tải trọng số encoder thành công.")
            else:
                print(f"Cảnh báo: Không nhận dạng được định dạng file '{pretrained_path}'.")

        else:
            print("Khởi tạo model từ đầu (random init).")

    # =======================================================================
    #    LOGIC FORWARD ĐÃ ĐƯỢC CẬP NHẬT (PHIÊN BẢN HIỆU QUẢ)
    # =======================================================================

    def perform_classification(self, support_features, support_mask, query_features, support_true_bg_mask=None):
        """
        Thực hiện phần so khớp prototype.
        Hàm này được tái sử dụng cho cả loss_1 và loss_2.
        Args:
            support_true_bg_mask (torch.Tensor, optional): Mask của vùng background thật.
        """
        # Downsample các mask về kích thước của feature map
        support_mask_downsampled = F.interpolate(support_mask, size=support_features.shape[2:], mode='bilinear')
        support_bg_mask_downsampled = 1.0 - support_mask_downsampled
        if support_true_bg_mask is not None:
            support_true_bg_mask_downsampled = F.interpolate(support_true_bg_mask, size=support_features.shape[2:], mode='nearest')
        else:
            support_true_bg_mask_downsampled = None
        
        # Tính score cho foreground và background bằng classifier
        fg_scores = self.classifier(
            qry_fts=query_features, 
            sup_fts=support_features, 
            sup_mask=support_mask_downsampled, 
            true_bg_mask=support_true_bg_mask_downsampled, # Truyền vào
            mode='gridconv+'
        )
        bg_scores = self.classifier(
            qry_fts=query_features, 
            sup_fts=support_features, 
            sup_mask=support_bg_mask_downsampled, 
            true_bg_mask=support_true_bg_mask_downsampled, # Truyền vào
            mode='gridconv'
        )
        
        # Ghép kết quả
        scores_low_res = torch.cat([bg_scores, fg_scores], dim=1) * self.logit_scale
        
        # Upsample kết quả về kích thước ảnh gốc (224x224)
        scores_upsampled = F.interpolate(scores_low_res, size=(224, 224), mode='bilinear', align_corners=False)
        return scores_upsampled

    def forward(self, support_image, support_mask, query_image, support_true_bg_mask):
        """
        Forward pass chính, nhận thêm support_true_bg_mask.
        """
        # --- 1. Trích xuất đặc trưng ---
        input_concat = torch.cat([support_image, query_image], dim=0)
        _, _, all_features = self.encoder(input_concat)
        support_features, query_features = torch.chunk(all_features, 2, dim=0)

        # --- 2. Tính toán so khớp cho loss_1 (Support -> Query) ---
        predicted_query_scores = self.perform_classification(
            support_features, 
            support_mask, 
            query_features,
            support_true_bg_mask # Truyền vào
        )

        # Trả về cả scores và features để train_oneshot.py có thể tái sử dụng
        return predicted_query_scores, support_features, query_features