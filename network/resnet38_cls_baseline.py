import torch
import torch.nn as nn
import torch.nn.functional as F
from network import resnet38d_baseline

class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.n_class = n_class
        
        # 1. Khởi tạo backbone baseline
        self.backbone = resnet38d_baseline.Net()

        # 2. Định nghĩa các đầu phân loại
        self.dropout7 = nn.Dropout2d(0.5)
        self.ic1 = nn.Conv2d(512, n_class, 1)      # Head cho b_45 (Scale 1: Low-level)
        self.ic2 = nn.Conv2d(1024, n_class, 1)     # Head cho b_52 (Scale 2: Mid-level)
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False) # Head cho conv6 (Scale 3: High-level)

        self.from_scratch_layers = [self.ic1, self.ic2, self.fc8]
        self._init_weights()

    def _init_weights(self):
        for m in self.from_scratch_layers:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, return_features=False):
        b_45, b_52, conv6 = self.backbone(x)
        
        # Nhánh 1 (Scale 1)
        out1_logits_map = self.ic1(b_45)
        x1 = F.avg_pool2d(out1_logits_map, kernel_size=(out1_logits_map.size(2), out1_logits_map.size(3)), padding=0).view(out1_logits_map.size(0), -1)

        # Nhánh 2 (Scale 2)
        out2_logits_map = self.ic2(b_52)
        x2 = F.avg_pool2d(out2_logits_map, kernel_size=(out2_logits_map.size(2), out2_logits_map.size(3)), padding=0).view(out2_logits_map.size(0), -1)

        # Nhánh 3 (Main Scale)
        x = self.dropout7(conv6)
        pooled_conv6 = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        feature = pooled_conv6.view(pooled_conv6.size(0), -1)
        x_logits = self.fc8(pooled_conv6).view(pooled_conv6.size(0), -1)
        
        y = torch.sigmoid(x_logits)
        
        if return_features:
            return x1, x2, x_logits, feature, y, conv6
        return x1, x2, x_logits, feature, y

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if 'backbone' in name:
                if 'weight' in name: groups[0].append(p)
                else: groups[1].append(p)
            else:
                if 'weight' in name: groups[2].append(p)
                else: groups[3].append(p)
        return groups

class Net_CAM(nn.Module):
    def __init__(self, n_class):
        super(Net_CAM, self).__init__()
        self.model = Net(n_class)

    def forward(self, x):
        # 1. Lấy feature maps từ backbone
        b_45, b_52, conv6 = self.model.backbone(x)

        # 2. Tạo CAM cho cả 3 scale
        # Scale 1: Low-level features (High resolution, noisy)
        cam1 = F.conv2d(b_45, self.model.ic1.weight)
        cam1 = F.relu(cam1)

        # Scale 2: Mid-level features
        cam2 = F.conv2d(b_52, self.model.ic2.weight)
        cam2 = F.relu(cam2)

        # Scale 3: High-level features (Low resolution, semantic)
        cam8 = F.conv2d(conv6, self.model.fc8.weight)
        cam8 = F.relu(cam8)

        # Trả về dictionary để dễ xử lý ở infer_fun
        return {
            "cam1": cam1,
            "cam2": cam2,
            "cam8": cam8,
            "features": conv6
        }

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

    def eval(self):
        return self.model.eval()

    def train(self, mode=True):
        return self.model.train(mode)