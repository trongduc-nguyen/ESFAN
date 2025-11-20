# File: joint_model.py (NÂNG CẤP VỚI ALIGNMENT LOSS & TRUE_BG_MASK)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.resnet38_cls_baseline import Net as ClassifierNet
from tool.alpmodule import MultiProtoAsConv
from network import resnet38d

class JointModel(nn.Module):
    def __init__(self, n_class=4, pretrained_path=None):
        super().__init__()
        
        # --- 1. Khởi tạo các thành phần ---
        self.classifier_head = ClassifierNet(n_class=n_class)
        self.encoder = self.classifier_head.backbone 
        self.oneshot_head = MultiProtoAsConv(proto_grid=(14, 14), feature_hw=(28, 28))
        self.logit_scale = nn.Parameter(torch.ones([]) * 20.0)

        # --- 2. Load trọng số pre-trained cho ENCODER ---
        if pretrained_path and os.path.exists(pretrained_path):
            if pretrained_path.endswith('.params'):
                print(f"Loading ImageNet weights for encoder from: {pretrained_path}")
                weights_dict = resnet38d.convert_mxnet_to_torch(pretrained_path)
                self.encoder.load_state_dict(weights_dict, strict=False)
                print("Encoder initialized with ImageNet weights.")
            elif pretrained_path.endswith('.pth'):
                print(f"Loading full model state from: {pretrained_path}")
                self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
                print("Resumed training from checkpoint.")

    def forward_classification(self, x,return_features=False):
        """Chạy forward pass cho nhánh classification."""
        return self.classifier_head(x, return_features=return_features)
    def perform_oneshot_classification(self, support_features, support_mask, query_features, support_true_bg_mask=None):
        """
        Thực hiện phần so khớp prototype.
        Hàm này được tái sử dụng cho cả loss_1 và loss_2.
        """
        support_mask_downsampled = F.interpolate(support_mask, size=support_features.shape[2:], mode='bilinear')
        support_bg_mask_downsampled = 1.0 - support_mask_downsampled
        
        if support_true_bg_mask is not None:
            support_true_bg_mask_downsampled = F.interpolate(support_true_bg_mask, size=support_features.shape[2:], mode='nearest')
        else:
            support_true_bg_mask_downsampled = None
        
        fg_scores = self.oneshot_head(
            qry_fts=query_features, sup_fts=support_features, sup_mask=support_mask_downsampled, 
            true_bg_mask=support_true_bg_mask_downsampled, mode='gridconv+'
        )
        bg_scores = self.oneshot_head(
            qry_fts=query_features, sup_fts=support_features, sup_mask=support_bg_mask_downsampled, 
            true_bg_mask=support_true_bg_mask_downsampled, mode='gridconv'
        )
        
        scores_low_res = torch.cat([bg_scores, fg_scores], dim=1) * self.logit_scale
        scores_upsampled = F.interpolate(scores_low_res, size=(224, 224), mode='bilinear', align_corners=False)
        return scores_upsampled

    def forward_oneshot(self, support_image, support_mask, query_image, support_true_bg_mask):
        """
        Chạy forward pass cho nhánh one-shot, trả về cả features để tính alignment loss.
        """
        # --- Trích xuất đặc trưng ---
        input_concat = torch.cat([support_image, query_image], dim=0)
        _, _, all_features = self.encoder(input_concat)
        support_features, query_features = torch.chunk(all_features, 2, dim=0)

        # --- Tính toán so khớp cho loss_1 (Support -> Query) ---
        predicted_query_scores = self.perform_oneshot_classification(
            support_features, support_mask, query_features, support_true_bg_mask
        )

        return predicted_query_scores, support_features, query_features

    def get_parameter_groups(self):
        """Sử dụng lại hàm get_parameter_groups của classifier để quản lý learning rate."""
        return self.classifier_head.get_parameter_groups()