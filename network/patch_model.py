import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_base_resnet18_cifar():
    """
    Hàm helper để khởi tạo và load weight cho backbone ResNet18 (CIFAR style).
    """
    print("Initializing ResNet18 Backbone (CIFAR-10 style)...")
    # 1. Khởi tạo model gốc
    model = timm.create_model("resnet18", pretrained=False)

    # 2. Override kiến trúc cho ảnh nhỏ (32x32)
    # Stride=1 giúp giữ kích thước feature map lớn (4x4) ở layer cuối
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10) 

    # 3. Load Pretrained Weights
    try:
        # Link weight CIFAR-10 chất lượng cao
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
            map_location="cpu", 
            file_name="resnet18_cifar10.pth",
        )
        model.load_state_dict(state_dict)
        print("-> Loaded pretrained weights from HuggingFace.")
    except Exception as e:
        print(f"Warning: Could not load pretrained weights. Error: {e}")
        print("Initializing with random weights.")

    return model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=49):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

class PatchModel(nn.Module):
    def __init__(self, n_class=4, device='cuda'):
        super(PatchModel, self).__init__()
        
        self.backbone = get_base_resnet18_cifar()
        self.backbone.fc = nn.Identity()
        
        self.pos_encoder = PositionalEncoding(d_model=512, max_len=49)
        
        # Transformer: batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # [NEW] Learnable Scale parameter (Init = 0)
        # Giúp model bắt đầu như một CNN thuần túy
        self.attn_scale = nn.Parameter(torch.zeros(1))
        
        self.num_classes = n_class 
        self.fc = nn.Linear(512, self.num_classes)
        
        self.proj_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        self.to(device)

    def forward(self, x, batch_size_img):
        # 1. CNN Feature
        feat_map = self.backbone.forward_features(x) # [B_tot, 512, 4, 4]
        feat_vec = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1) # [B_tot, 512]
        
        # 2. Self-Attention với Residual Connection an toàn
        feat_seq = feat_vec.view(batch_size_img, 49, 512)
        
        # Thêm Positional Encoding
        feat_seq_pe = self.pos_encoder(feat_seq)
        
        # Qua Transformer
        attn_out = self.transformer(feat_seq_pe)
        
        # [LOGIC MỚI] Residual + Zero Init Scaling
        # Ban đầu attn_scale = 0 -> feat_enriched = feat_seq (CNN gốc)
        # Sau đó attn_scale học tăng dần -> Attention bắt đầu có tác dụng
        feat_enriched_seq = feat_seq + self.attn_scale * attn_out
        
        # Flatten
        feat_enriched_flat = feat_enriched_seq.view(-1, 512)
        
        # 3. Heads
        logits = self.fc(feat_enriched_flat)
        embedding = self.proj_head(feat_enriched_flat)
        embedding = F.normalize(embedding, dim=1)
        
        return logits, embedding, feat_enriched_seq

# Hàm wrapper để tương thích code cũ nếu cần
def get_patch_model(n_class=4, device='cuda'):
    return PatchModel(n_class=n_class, device=device)

if __name__ == "__main__":
    # Test model structure
    try:
        from torchinfo import summary
        model = get_patch_model(n_class=4, device='cpu')
        print("Checking PatchModel Architecture (CNN + Transformer)...")
        
        # Giả lập input: Batch=2 ảnh, mỗi ảnh 49 patch -> Tổng 98 patch
        dummy_input = torch.randn(98, 3, 32, 32)
        batch_size_img = 2
        
        logits, embed, feat_seq = model(dummy_input, batch_size_img)
        
        print(f"\nLogits shape: {logits.shape} (Expected: [98, 5])")
        print(f"Embedding shape: {embed.shape} (Expected: [98, 128])")
        print(f"Fused Feature shape: {feat_seq.shape} (Expected: [2, 49, 512])")
        
    except ImportError:
        print("Please install 'torchinfo' to visualize summary.")