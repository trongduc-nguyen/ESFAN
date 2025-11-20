# network/resnet38_cls_baseline.py
# PHIÊN BẢN CẬP NHẬT - Phản ánh chính xác logic từ file resnet38_cls.py gốc

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import backbone baseline của chúng ta, KHÔNG phải resnet38d gốc
from network import resnet38d_baseline

class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.n_class = n_class
        
        # 1. Khởi tạo backbone baseline
        self.backbone = resnet38d_baseline.Net()

        # 2. Định nghĩa các đầu phân loại và các lớp khác y hệt file gốc
        self.dropout7 = nn.Dropout2d(0.5)
        self.ic1 = nn.Conv2d(512, n_class, 1)      # Head cho b_45
        self.ic2 = nn.Conv2d(1024, n_class, 1)     # Head cho b_52
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False) # Head cho conv6

        # Các lớp này sẽ được khởi tạo từ đầu (không dùng pre-trained)
        self.from_scratch_layers = [self.ic1, self.ic2, self.fc8]
        
        # Khởi tạo trọng số cho các lớp mới
        self._init_weights()

    def _init_weights(self):
        for m in self.from_scratch_layers:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x,  return_features=False):
        # 1. Lấy feature maps từ backbone baseline
        b_45, b_52, conv6 = self.backbone(x)
        
        # 2. Xử lý deep supervision cho các nhánh phụ
        # Nhánh 1 (từ b_45)
        out1_logits_map = self.ic1(b_45)
        x1 = F.avg_pool2d(out1_logits_map, kernel_size=(out1_logits_map.size(2), out1_logits_map.size(3)), padding=0)
        x1 = x1.view(x1.size(0), -1)

        # Nhánh 2 (từ b_52)
        out2_logits_map = self.ic2(b_52)
        x2 = F.avg_pool2d(out2_logits_map, kernel_size=(out2_logits_map.size(2), out2_logits_map.size(3)), padding=0)
        x2 = x2.view(x2.size(0), -1)

        # 3. Xử lý nhánh chính (từ conv6)
        x = self.dropout7(conv6)
        pooled_conv6 = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        
        feature = pooled_conv6.view(pooled_conv6.size(0), -1) # Vector feature [batch, 4096]
        
        x_logits = self.fc8(pooled_conv6)
        x_logits = x_logits.view(x_logits.size(0), -1)
        
        # Output cuối cùng để tính accuracy
        y = torch.sigmoid(x_logits)
        if return_features:
            return x1, x2, x_logits, feature, y, conv6 # Thêm conv6 vào
        return x1, x2, x_logits, feature, y
        # return x1, x2, x_logits, feature, y

    def get_parameter_groups(self):
        """
        Phân chia các tham số thành các nhóm để áp dụng learning rate khác nhau.
        Logic này được sao chép từ file gốc để đảm bảo tính tương thích.
        """
        groups = ([], [], [], []) # weight_decay, no_weight_decay, scratch_weight_decay, scratch_no_weight_decay
        
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            is_from_scratch = any(id(p) == id(layer_p) for layer in self.from_scratch_layers for layer_p in layer.parameters())

            if 'backbone' in name:
                if 'weight' in name:
                    groups[0].append(p)
                else:
                    groups[1].append(p)
            else: # classification heads (from_scratch)
                if 'weight' in name:
                    groups[2].append(p)
                else:
                    groups[3].append(p)
        return groups
class Net_CAM(nn.Module):
    def __init__(self, n_class):
        super(Net_CAM, self).__init__()
        # 1. Khởi tạo kiến trúc model Net gốc
        #    Kiến trúc này chứa cả backbone và các đầu phân loại
        self.model = Net(n_class)

    def forward(self, x):
        # 1. Lấy feature map cuối cùng từ backbone bên trong self.model
        _, _, conv6 = self.model.backbone(x)

        # 2. Lấy trọng số từ đầu phân loại cuối cùng bên trong self.model
        #    Trọng số có shape [n_class, 4096, 1, 1]
        cam_weights = self.model.fc8.weight

        # 3. Tạo CAM bằng tích chập F.conv2d
        #    Input: conv6 [batch, 4096, H, W]
        #    Weight: cam_weights [n_class, 4096, 1, 1]
        #    Output: cam [batch, n_class, H, W]
        cam = F.conv2d(conv6, cam_weights)
        cam = F.relu(cam) # Áp dụng ReLU như thông lệ

        return cam

    # Hàm load state_dict vẫn giữ nguyên
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

    # Hàm eval để chuyển model con sang chế độ evaluation
    def eval(self):
        return self.model.eval()

    # Hàm train để chuyển model con sang chế độ training (nếu cần)
    def train(self, mode=True):
        return self.model.train(mode)