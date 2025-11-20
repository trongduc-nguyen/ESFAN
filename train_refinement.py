import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import random
from collections import defaultdict

# ===================================================================================
# THÀNH PHẦN 1: DATASET THÔNG MINH
# ===================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class PairedDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        super(PairedDataset, self).__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
        
        print("Đang quét dataset để nhóm ảnh theo lớp...")
        class_to_images = defaultdict(list)
        image_to_classes = defaultdict(set)
        
        for img_name in tqdm(all_images, desc="Scanning labels"):
            label_path = os.path.join(label_dir, img_name)
            if os.path.exists(label_path):
                label = np.array(Image.open(label_path))
                unique_classes = np.unique(label)
                for cls in unique_classes:
                    if cls != 0: # Bỏ qua background có giá trị 0 (đen)
                        class_to_images[cls].append(img_name)
                        image_to_classes[img_name].add(cls)

        self.valid_classes = {cls for cls, imgs in class_to_images.items() if len(imgs) >= 2}
        print(f"Các lớp hợp lệ có thể tạo cặp (xuất hiện trong >= 2 ảnh): {self.valid_classes}")

        self.image_list = [
            img for img in all_images 
            if any(cls in self.valid_classes for cls in image_to_classes[img])
        ]
        
        self.class_to_images = {
            cls: imgs for cls, imgs in class_to_images.items() if cls in self.valid_classes
        }

        self.image_to_classes = image_to_classes

        if not self.image_list:
            raise RuntimeError("Không tìm thấy ảnh nào có chứa các lớp hợp lệ để tạo cặp.")

        print(f"Quét xong! {len(self.image_list)} ảnh hợp lệ có thể được dùng để tạo cặp.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img1_name = self.image_list[index]
        available_classes = self.image_to_classes[img1_name].intersection(self.valid_classes)
        
        if not available_classes:
            class_A = random.choice(list(self.valid_classes))
            img1_name = random.choice(self.class_to_images[class_A])
        else:
            class_A = random.choice(list(available_classes))
        
        possible_img2 = self.class_to_images[class_A]
        img2_name = random.choice(possible_img2)
        if len(possible_img2) > 1:
            while img2_name == img1_name:
                img2_name = random.choice(possible_img2)
            
        img1 = Image.open(os.path.join(self.img_dir, img1_name)).convert("RGB")
        pl1 = Image.open(os.path.join(self.label_dir, img1_name))
        
        img2 = Image.open(os.path.join(self.img_dir, img2_name)).convert("RGB")
        pl2 = Image.open(os.path.join(self.label_dir, img2_name))
        
        if self.transform:
            img1 = self.transform(img1)
            pl1 = torch.from_numpy(np.array(pl1)).long()
            img2 = self.transform(img2)
            pl2 = torch.from_numpy(np.array(pl2)).long()

        return img1, pl1, img2, pl2, torch.tensor(class_A).long()


# ===================================================================================
# THÀNH PHẦN 2: MODULE TÍNH TOÁN CỐT LÕI
# ===================================================================================
try:
    # Sử dụng backbone mới đã được tạo ở Bước 1
    from resnet38d_refined import Net as RefinedBackbone
except ImportError:
    print("LỖI: Hãy tạo file 'network/resnet38d_refined.py' theo hướng dẫn.")
    exit()

class AttentionSimilarityModule(nn.Module):
    def __init__(self):
        super(AttentionSimilarityModule, self).__init__()

    # <<< THAY ĐỔI: THÊM LẠI `pl_support` ĐỂ LỌC BG DỄ >>>
    def forward(self, f_query, f_support, mask_support, pl_support):
        f_q_vecs = f_query.squeeze(0).view(f_query.shape[1], -1)
        f_s_vecs = f_support.squeeze(0).view(f_support.shape[1], -1)
        
        mask_s_vecs = mask_support.squeeze(0).view(-1)
        pl_s_vecs = pl_support.squeeze(0).view(-1)
        
        f_q_vecs_norm = F.normalize(f_q_vecs, p=2, dim=0)
        f_s_vecs_norm = F.normalize(f_s_vecs, p=2, dim=0)
        
        sim = torch.matmul(f_q_vecs_norm.t(), f_s_vecs_norm)

        # --- Xử lý FG (Không đổi) ---
        sim_fg = sim.masked_fill(mask_s_vecs.unsqueeze(0) == 0, -1e9)
        attn_fg = F.softmax(sim_fg, dim=1)
        score_fg = torch.sum(attn_fg * sim, dim=1)

        # <<< THAY ĐỔI: BỔ SUNG LẠI LOGIC "HARD BACKGROUND MINING" >>>
        # BG khó là những pixel không phải FG (mask_s_vecs == 0)
        # VÀ cũng không phải BG dễ (pl_s_vecs != 0)
        hard_bg_indices = (mask_s_vecs == 0) & (pl_s_vecs != 0)

        # Xử lý các trường hợp đặc biệt
        if not torch.any(mask_s_vecs == 1): # Nếu không có pixel FG nào
            score_fg.fill_(-1.0) # Điểm FG rất thấp
            # Nếu không có FG, tất cả các pixel khác đều là BG
            score_bg = torch.full_like(score_fg, 1.0) # Điểm BG rất cao
        elif not torch.any(hard_bg_indices): # Nếu không có pixel BG khó nào
            score_fg.fill_(1.0) # Điểm FG rất cao
            # Nếu không có BG khó, tất cả các pixel khác đều là FG
            score_bg = torch.full_like(score_fg, -1.0) # Điểm BG rất thấp
        else:
            # --- Xử lý BG khó (Logic mới) ---
            sim_bg = sim.masked_fill(~hard_bg_indices.unsqueeze(0), -1e9)
            attn_bg = F.softmax(sim_bg, dim=1)
            score_bg = torch.sum(attn_bg * sim, dim=1)

        # Xếp chồng điểm số BG khó và điểm số FG
        pred_map = torch.stack([score_bg, score_fg], dim=1)
        pred_map = pred_map.view(1, f_query.shape[2], f_query.shape[3], 2).permute(0, 3, 1, 2)
        
        return pred_map

# ===================================================================================
# THÀNH PHẦN 3: SCRIPT HUẤN LUYỆN CHÍNH (ĐÃ SỬA LẠI)
# ===================================================================================

def train_refinement_phase(args):
    # --- 1. Setup ---
    backbone = RefinedBackbone()
    print(f"Đang tải trọng số backbone từ: {args.weights}")
    
    state_dict_from_file = torch.load(args.weights, map_location='cpu')
    backbone_state_dict = {}
    for key, value in state_dict_from_file.items():
        if key.startswith('backbone.'):
            new_key = key[len('backbone.'):]
            backbone_state_dict[new_key] = value
            
    backbone.load_state_dict(backbone_state_dict, strict=True)
    backbone = backbone.cuda()
    
    attn_module = AttentionSimilarityModule().cuda()
    
    transform_train = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor()
    ])
    dataset = PairedDataset(img_dir=args.img_dir, label_dir=args.label_dir, 
                            transform=transform_train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Quay lại dùng CrossEntropyLoss vì đầu ra là [score_bg, score_fg]
    
    # --- 2. Vòng lặp huấn luyện ---
    for epoch in range(args.max_epoches):
        backbone.train()
        total_loss = 0.0
        processed_items = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.max_epoches}")
        for i, data in enumerate(progress_bar):
            try:
                img1, pl1, img2, pl2, class_A = data
                img1, pl1, img2, pl2, class_A = \
                    img1.cuda(), pl1.cuda(), img2.cuda(), pl2.cuda(), class_A.cuda()
                
                _, _, f1 = backbone(img1)
                _, _, f2 = backbone(img2)
                
                mask1_binary = (pl1 == class_A).long()
                mask2_binary = (pl2 == class_A).long()
                
                size_ds = f1.shape[2:]
                mask1_binary_ds = F.interpolate(mask1_binary.unsqueeze(1).float(), size=size_ds, mode='nearest').squeeze(1).long()
                mask2_binary_ds = F.interpolate(mask2_binary.unsqueeze(1).float(), size=size_ds, mode='nearest').squeeze(1).long()
                pl1_ds = F.interpolate(pl1.unsqueeze(1).float(), size=size_ds, mode='nearest').squeeze(1).long()
                pl2_ds = F.interpolate(pl2.unsqueeze(1).float(), size=size_ds, mode='nearest').squeeze(1).long()
                
                # --- Chiều thuận: 1 -> 2 ---
                pred12 = attn_module(f_query=f2, f_support=f1, mask_support=mask1_binary_ds, pl_support=pl1_ds)
                loss1 = criterion(pred12, mask2_binary_ds)
                
                # --- Chiều nghịch: 2 -> 1 ---
                pred21 = attn_module(f_query=f1, f_support=f2, mask_support=mask2_binary_ds, pl_support=pl2_ds)
                loss2 = criterion(pred21, mask1_binary_ds)
                
                loss = loss1 + args.alpha * loss2
                
                if torch.isnan(loss) or torch.isinf(loss):
                    tqdm.write(f"Cảnh báo: Bỏ qua bước {i} do loss là NaN hoặc Inf.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                processed_items += 1
                
                if processed_items > 0:
                    progress_bar.set_postfix({'Loss': total_loss / processed_items})

            except Exception as e:
                tqdm.write(f"LỖI: Gặp lỗi ở bước {i}. Bỏ qua. Chi tiết: {e}")
                continue
        if epoch % 3 == 0:
            save_path = os.path.join(args.save_folder, f'fewshot_refined_backbone_ep{epoch+1}.pth')
            torch.save(backbone.state_dict(), save_path)
            print(f"Đã lưu checkpoint tinh chỉnh tại: {save_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Giai đoạn 2: Tinh chỉnh Backbone.")
    
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn đến checkpoint backbone đã huấn luyện ở GĐ1.")
    parser.add_argument("--img_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh huấn luyện gốc.")
    parser.add_argument("--label_dir", type=str, required=True, help="Đường dẫn đến thư mục pseudo-label đã được làm sạch.")
    parser.add_argument("--save_folder", type=str, required=True, help="Thư mục để lưu các checkpoint đã tinh chỉnh.")
    
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--max_epoches", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.01, help="Hệ số cho loss đối xứng.")
    parser.add_argument("--crop_size", type=int, default=256, help="Kích thước ảnh sau khi resize để huấn luyện.")
    
    # Tham số này không còn được dùng trực tiếp nhưng giữ lại để không gây lỗi
    parser.add_argument("--bg_threshold", type=int, default=220, help="Ngưỡng độ sáng để xác định background.") 
    
    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    
    train_refinement_phase(args)