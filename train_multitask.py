import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
from skimage import morphology

# Import các module cần thiết
from network.multitask_model import MultiTaskNet
from tool.GenDataset import Stage1_TrainDataset
from torchvision import transforms
from tool import torchutils, pyutils
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# --- HÀM TIỆN ÍCH CHO BACKGROUND MASK ---
def gen_bg_mask_from_pil(img_pil):
    """
    Tạo mask nhị phân cho background từ ảnh PIL.
    Trả về một mảng boolean (True ở vùng background).
    """

    orig_img_np = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    binary_mask = (binary == 255)
    bg_binary_mask = morphology.remove_small_objects(binary_mask, min_size=50, connectivity=1)
    return bg_binary_mask

# --- LỚP DATASET WRAPPER MỚI ---
class MultiTaskDataset(Dataset):
    def __init__(self, base_dataset, transform_support, transform_query, normalize_transform):
        self.base_dataset = base_dataset
        self.transform_support = transform_support
        self.transform_query = transform_query
        self.normalize_transform = normalize_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img_name, img_pil, label = self.base_dataset[index]
        
        # Tạo 2 views PIL đã được augment hình học
        support_img_pil = self.transform_support(img_pil)
        query_img_pil = self.transform_query(img_pil)
        
        # Tạo bg_mask từ support view (trước khi normalize)
        bg_mask = gen_bg_mask_from_pil(support_img_pil)
        
        # Chuyển các views sang tensor và normalize
        support_view_tensor = self.normalize_transform(support_img_pil)
        query_view_tensor = self.normalize_transform(query_img_pil)
        
        return {
            'img_name': img_name,
            'support_view': support_view_tensor,
            'query_view': query_view_tensor,
            'label': label,
            'bg_mask': torch.from_numpy(bg_mask) # Chuyển bg_mask sang Tensor
        }

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Multi-Task đồng thời Classification và Segmentation Consistency")
    
    # --- Tham số chung ---
    parser.add_argument("--dataroot", default='LUAD-HistoSeg/training/img/', type=str,
                        help="Thư mục training (chứa ảnh).")
    parser.add_argument("--n_class", default=4, type=int, help="Số lớp foreground.")
    parser.add_argument("--dataset", default="luad", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--output_dir", default='checkpoints_mutitask', type=str)

    # --- Tham số Model ---
    parser.add_argument("--weights", type=str, default=None,
                        help="(Tùy chọn) Đường dẫn đến checkpoint Giai đoạn 1 để bắt đầu.")
    parser.add_argument("--proto_grid_size", type=int, nargs=2, default=[14, 14],
                        help="Kích thước lưới cho local prototypes.")

    # --- Tham số Training ---
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--lambda_seg", default=0.01, type=float,
                        help="Hệ số cho loss segmentation consistency.")
    # Bỏ bg_threshold vì không còn dùng ngưỡng trên CAM
    # parser.add_argument("--bg_threshold", default=0.2, type=float) 

    args = parser.parse_args()
    print("Các tham số được sử dụng:"); print(vars(args))

    # --- 1. Model và Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskNet(n_class=args.n_class, proto_grid_size=tuple(args.proto_grid_size))
    
    if args.weights:
        print(f"Loading pre-trained weights from: {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)

    model.to(device)
    
    # Ước tính số ảnh train của LUAD, bạn có thể thay đổi con số này
    num_train_images = 16678 
    max_step = (num_train_images // args.batch_size) * args.epochs
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    # --- 2. Data Augmentation và Dataset ---
    transform_support = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])
    
    transform_query = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ])

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset_base = Stage1_TrainDataset(data_path=args.dataroot, dataset=args.dataset, transform=None)
    
    train_dataset_multitask = MultiTaskDataset(
        base_dataset=train_dataset_base,
        transform_support=transform_support,
        transform_query=transform_query,
        normalize_transform=normalize_transform
    )
    
    train_loader = DataLoader(train_dataset_multitask,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    
    # --- 3. Loss Functions ---
    loss_cls_fn = nn.MultiLabelSoftMarginLoss()
    loss_seg_fn = nn.CrossEntropyLoss(ignore_index=4)

    # --- 4. Vòng lặp Training ---
    print("\n--- Bắt đầu quá trình Huấn luyện Multi-Task (với Heuristic BG) ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        avg_meter = pyutils.AverageMeter('loss_total', 'loss_cls', 'loss_seg')
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            support_imgs_tensor = batch['support_view'].to(device)
            query_imgs_tensor = batch['query_view'].to(device)
            labels = batch['label'].to(device)
            bg_masks_tensor = batch['bg_mask'].to(device)
            
            optimizer.zero_grad()
            
            x1, x2, x3, cam_support = model(support_imgs_tensor)
            
            with torch.no_grad():
                cam_support_resized = F.interpolate(cam_support, size=support_imgs_tensor.shape[-2:],
                                                    mode='bilinear', align_corners=False).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                cam_filtered = cam_support_resized * labels_np[:, :, np.newaxis, np.newaxis]
                
                foreground_mask = np.argmax(cam_filtered, axis=1)
                pseudo_mask_support = torch.from_numpy(foreground_mask).to(device)
                
                pseudo_mask_support[bg_masks_tensor] = 4
            
            _, _, _, _, seg_logits = model(support_imgs_tensor, 
                                           query_imgs_tensor, 
                                           pseudo_mask_support)

            loss_cls1 = loss_cls_fn(x1, labels)
            loss_cls2 = loss_cls_fn(x2, labels)
            loss_cls3 = loss_cls_fn(x3, labels)
            loss_cls = 0.2 * loss_cls1 + 0.3 * loss_cls2 + 0.5 * loss_cls3
            
            seg_logits_resized = F.interpolate(seg_logits, size=support_imgs_tensor.shape[-2:],
                                               mode='bilinear', align_corners=False)
            loss_seg = loss_seg_fn(seg_logits_resized, pseudo_mask_support)

            loss_total = loss_cls + args.lambda_seg * loss_seg

            loss_total.backward()
            optimizer.step()
            
            avg_meter.add({'loss_total': loss_total.item(), 'loss_cls': loss_cls.item(), 'loss_seg': loss_seg.item()})
            progress_bar.set_postfix({'Loss': avg_meter.get('loss_total'), 'L_cls': avg_meter.get('loss_cls'), 'L_seg': avg_meter.get('loss_seg')})
            
        print(f"Epoch {epoch+1} summary: Total Loss={avg_meter.get('loss_total'):.4f}, "
              f"Cls Loss={avg_meter.get('loss_cls'):.4f}, Seg Loss={avg_meter.get('loss_seg'):.4f}")

        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            print(f"Đang lưu checkpoint tại epoch {epoch+1}...")
            save_path = os.path.join(args.output_dir, f"multitask_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Đã lưu checkpoint tại: {save_path}")

    print("\nTraining hoàn tất.")


if __name__ == '__main__':
    main()