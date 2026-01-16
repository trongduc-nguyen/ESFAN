import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchStitcher(nn.Module):
    """
    Module chuyển đổi chuỗi Feature/Logits từ Patch Model thành Feature Map lớn.
    
    Phiên bản mới hỗ trợ output từ Transformer/FC:
    Input:  Sequence [B, 49, C] hoặc Flat [B*49, C]
    Output: Feature Map [B, C, 28, 28] (Khớp với S1)
    """
    def __init__(self, grid_size=7, target_size=28):
        super(PatchStitcher, self).__init__()
        self.grid_size = grid_size
        self.target_size = target_size
        
    def forward(self, x, batch_size=None):
        """
        Args:
            x: Input tensor. Có thể là:
               - [B, 49, C]: Sequence từ Transformer
               - [B*49, C]: Logits phẳng từ FC
            batch_size: Cần thiết nếu input là dạng phẳng [B*49, C] để reshape.
        """
        # Trường hợp 1: Input dạng phẳng [N_total, C] (thường là Logits)
        if x.dim() == 2:
            if batch_size is None:
                # Nếu không truyền batch_size, thử suy luận từ grid_size
                # B = N_total // 49
                batch_size = x.size(0) // (self.grid_size * self.grid_size)
            
            C = x.size(1)
            # Reshape về [B, 7, 7, C]
            x = x.view(batch_size, self.grid_size, self.grid_size, C)
        
        # Trường hợp 2: Input dạng Sequence [B, 49, C] (thường là Feature từ Transformer)
        elif x.dim() == 3:
            B, N, C = x.shape
            # Reshape về [B, 7, 7, C]
            x = x.view(B, self.grid_size, self.grid_size, C)
            
        # Đến đây x có dạng [B, 7, 7, C]
        # Permute về dạng ảnh: [B, C, 7, 7]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Upsample lên kích thước đích (28x28)
        # Dùng Bilinear để làm mượt thông tin giữa các patch lân cận
        x = F.interpolate(x, size=(self.target_size, self.target_size), 
                          mode='bilinear', align_corners=False)
        return x

class SemanticInjectedFusionBlock(nn.Module):
    """
    Fusion Block cải tiến: 
    Input: Main Feature (S1) + Patch Feature (Context) + Patch Semantic Logits (Guide)
    """
    def __init__(self, main_channels, patch_channels=512, num_classes=4):
        super(SemanticInjectedFusionBlock, self).__init__()
        
        # 1. Adapter cho Patch Feature (Align distribution)
        self.feat_adapter = nn.Sequential(
            nn.Conv2d(patch_channels, patch_channels, 1, bias=False),
            nn.BatchNorm2d(patch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Adapter cho Semantic Logits (Quan trọng để S1 hiểu ý nghĩa xác suất)
        self.sem_adapter = nn.Sequential(
            nn.Conv2d(num_classes, 64, 1, bias=False), # Map 5 class sang 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Tổng channel đầu vào = S1 + Patch(512) + Semantics(64)
        concat_dim = main_channels + patch_channels + 64
        
        # 3. SE-Block (Attention) để cân bằng 3 luồng thông tin
        # Giúp mạng tự học nên tin vào Global (S1) hay Local (Patch) ở từng vị trí
        self.se_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(concat_dim, concat_dim // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(concat_dim // 16, concat_dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 4. Project về dimension gốc của S1 (để không phá vỡ kiến trúc phía sau)
        self.project = nn.Sequential(
            nn.Conv2d(concat_dim, main_channels, 1, bias=False),
            nn.BatchNorm2d(main_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, main_feat, patch_feat, patch_logits_map):
        """
        Args:
            main_feat: [B, C_main, 28, 28]
            patch_feat: [B, 512, 28, 28]
            patch_logits_map: [B, 5, 28, 28]
        """
        # Align features
        patch_feat_adapted = self.feat_adapter(patch_feat)
        sem_feat_adapted = self.sem_adapter(patch_logits_map)
        
        # Concatenate 3 thành phần
        cat_feat = torch.cat([main_feat, patch_feat_adapted, sem_feat_adapted], dim=1)
        
        # Recalibration (SE Attention)
        w = self.se_fc(cat_feat)
        cat_feat = cat_feat * w
        
        # Project back to original dimension
        out = self.project(cat_feat)
        
        return out