import torch
import torch.nn as nn
import torch.nn.functional as F
from network import resnet38d_baseline
# Import module fusion mới (Giả sử bạn lưu ở network/fusion_modules.py)
from network.fusion_modules import PatchStitcher, SemanticInjectedFusionBlock

# ==============================================================================
# MAIN NETWORK (S1 - Global Student)
# ==============================================================================

class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.n_class = n_class
        
        # 1. Khởi tạo backbone baseline (ResNet38d)
        self.backbone = resnet38d_baseline.Net()

        # 2. Định nghĩa các đầu phân loại (Classification Heads)
        self.dropout7 = nn.Dropout2d(0.5)
        
        # Deep Supervision Heads
        self.ic1 = nn.Conv2d(512, n_class, 1)      # Head cho b_45 (Low level)
        self.ic2 = nn.Conv2d(1024, n_class, 1)     # Head cho b_52 (Mid level)
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False) # Head cho conv6 (High level)

        # --- FUSION MODULES (CẬP NHẬT MỚI) ---
        # Stitcher dùng để ghép feature map và logits map từ patch
        # Grid 7x7, Patch Feat 4x4 => Output 28x28
        self.stitcher = PatchStitcher()
        
        # Semantic Injected Fusion Blocks
        # Input: Main Feat + Patch Feat (512) + Patch Logits (n_class + 1)
        # Output: Main Feat Channels (để giữ nguyên kiến trúc S1 phía sau)
        num_patch_classes = n_class
        
        self.fuse_low = SemanticInjectedFusionBlock(main_channels=512, 
                                                    patch_channels=512, 
                                                    num_classes=num_patch_classes)
                                                    
        self.fuse_mid = SemanticInjectedFusionBlock(main_channels=1024, 
                                                    patch_channels=512, 
                                                    num_classes=num_patch_classes)
                                                    
        self.fuse_high = SemanticInjectedFusionBlock(main_channels=4096, 
                                                     patch_channels=512, 
                                                     num_classes=num_patch_classes)

        # Các lớp cần train với learning rate cao hơn (mới khởi tạo)
        self.from_scratch_layers = [self.ic1, self.ic2, self.fc8, 
                                    self.fuse_low, self.fuse_mid, self.fuse_high]
        
        self._init_weights()

    def _init_weights(self):
        for m in self.from_scratch_layers:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, patch_features_seq=None, patch_logits=None, return_features=False):
        """
        patch_features_seq: [B, 49, 512] (Feature đã qua Attention)
        patch_logits: [B*49, 5] (Logits thô)
        """
        b_45, b_52, conv6 = self.backbone(x)

        if patch_features_seq is not None and patch_logits is not None:
            B = x.size(0)
            
            # 1. Stitch Feature Map (Từ Attention Vector)
            # Input: [B, 49, 512] -> Output: [B, 512, 28, 28]
            patch_map = self.stitcher(patch_features_seq)
            
            # 2. Stitch Semantic Map
            # Input Logits: [B*49, 5] -> Reshape -> [B, 49, 5]
            patch_logits_reshaped = patch_logits.view(B, 49, -1)
            sem_map = self.stitcher(patch_logits_reshaped)
            
            # 3. Inject
            b_45 = self.fuse_low(b_45, patch_map, sem_map)
            b_52 = self.fuse_mid(b_52, patch_map, sem_map)
            conv6 = self.fuse_high(conv6, patch_map, sem_map)

        # 3. Classification Heads (Deep Supervision)
        
        # Nhánh 1 (Low)
        out1_logits_map = self.ic1(b_45)
        # GAP -> Vector
        x1 = F.avg_pool2d(out1_logits_map, kernel_size=(out1_logits_map.size(2), out1_logits_map.size(3)), padding=0).view(out1_logits_map.size(0), -1)

        # Nhánh 2 (Mid)
        out2_logits_map = self.ic2(b_52)
        x2 = F.avg_pool2d(out2_logits_map, kernel_size=(out2_logits_map.size(2), out2_logits_map.size(3)), padding=0).view(out2_logits_map.size(0), -1)

        # Nhánh 3 (High - Main)
        x_drop = self.dropout7(conv6)
        
        # Logits Map cho CAM (Dùng sau này để sinh mask)
        x_logits_map = self.fc8(conv6) # [B, 4, 28, 28]
        
        # Global Pooling cho Classification Prediction
        pooled_conv6 = F.avg_pool2d(x_drop, kernel_size=(x_drop.size(2), x_drop.size(3)), padding=0)
        feature = pooled_conv6.view(pooled_conv6.size(0), -1)
        
        x_logits = self.fc8(pooled_conv6).view(pooled_conv6.size(0), -1)
        y = torch.sigmoid(x_logits)
        
        if return_features:
            # Trả về tất cả để phục vụ Visualize và Debug
            # Lưu ý: b_45 ở đây là feature ĐÃ FUSE (nếu có fusion)
            return x1, x2, x_logits, feature, y, conv6, b_45

        return x1, x2, x_logits, feature, y

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            
            # Kiểm tra xem param thuộc module nào (Backbone hay New Layers)
            is_scratch = False
            for layer in self.from_scratch_layers:
                for lp in layer.parameters():
                    if id(p) == id(lp):
                        is_scratch = True
                        break
                if is_scratch: break
            
            if not is_scratch:
                # Backbone (Learning rate thấp - Groups 0 & 1)
                if 'weight' in name: groups[0].append(p)
                else: groups[1].append(p)
            else:
                # Heads + Fusion Modules (Learning rate cao - Groups 2 & 3)
                if 'weight' in name: groups[2].append(p)
                else: groups[3].append(p)
        return groups

class Net_CAM(nn.Module):
    def __init__(self, n_class):
        super(Net_CAM, self).__init__()
        self.model = Net(n_class)

    def forward(self, x, patch_features=None, patch_logits=None):
        # Wrapper để gọi model chính
        # Nếu đang train joint, cần truyền patch_features và patch_logits
        return self.model(x, patch_features, patch_logits)

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

    def eval(self):
        return self.model.eval()

    def train(self, mode=True):
        return self.model.train(mode)