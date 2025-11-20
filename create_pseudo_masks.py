# create_pseudo_masks.py
# Phiên bản này tạo nhãn giả "dày đặc" (dense).
# Mọi pixel sẽ được gán cho lớp foreground có điểm CAM cao nhất.

import argparse
import importlib
import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# Sử dụng lại Dataset gốc của tác giả để đọc label từ tên file
from tool.GenDataset import Stage1_TrainDataset
from torchvision import transforms

def create_dense_masks(args):
    """
    Tạo nhãn giả "dày đặc", trong đó mỗi pixel được gán cho lớp foreground
    có điểm CAM cao nhất.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo model Net_CAM (để tạo CAM)
    model_module = importlib.import_module(args.network)
    model = getattr(model_module, 'Net_CAM')(n_class=args.n_class)
    
    # 2. Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Load Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform, dataset=args.dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Chuẩn bị thư mục lưu trữ và palette
    os.makedirs(args.mask_save_path, exist_ok=True)
    # Palette này vẫn hữu ích để visualize, dù không có lớp background 4
    PALETTE_DATA = [205, 51, 51, 0, 255, 0, 65, 105, 225, 255, 165, 0, 255, 255, 255]
    
    print(f"\n--- Bắt đầu tạo Pseudo-Masks (Dense method, không có background) ---")
    print(f"Masks sẽ được lưu tại: {args.mask_save_path}")

    # 5. Vòng lặp tạo mask
    for img_names, imgs, labels in tqdm(data_loader, desc="Đang tạo nhãn giả"):
        imgs = imgs.to(device)

        with torch.no_grad():
            cams_batch = model(imgs)
            cams_batch_resized = F.interpolate(cams_batch, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
        
        cams_np = cams_batch_resized.cpu().numpy()
        labels_np = labels.numpy()

        for i in range(len(img_names)):
            img_name = img_names[i]
            cam_single = cams_np[i]
            label_single = labels_np[i]
            
            # --- LOGIC TẠO PSEUDO-MASK "DENSE" ---
            
            # a. Áp dụng image-level label để loại bỏ các kênh không liên quan.
            #    Với các kênh không có trong ảnh, gán điểm số rất thấp (-infinity)
            #    để đảm bảo chúng không bao giờ được chọn bởi argmax.
            cam_filtered = cam_single.copy()
            for c in range(args.n_class):
                if label_single[c] == 0:
                    cam_filtered[c, :, :] = -np.inf
            
            # b. Tìm lớp có điểm số cao nhất cho mỗi pixel.
            #    `argmax` sẽ trả về giá trị từ 0 đến 3.
            final_mask = np.argmax(cam_filtered, axis=0)
            
            # ------------------------------------
            
            # c. Lưu mask
            mask_img = Image.fromarray(final_mask.astype(np.uint8), "P")
            mask_img.putpalette(PALETTE_DATA)
            mask_img.save(os.path.join(args.mask_save_path, img_name + '.png'), format='PNG')

    print("\n--- Hoàn tất tạo Pseudo-Masks ---")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Tạo Nhãn giả (Pseudo-Masks) 'dày đặc' từ model Giai đoạn 1")
    
    # # Các tham số cần thiết
    # parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str)
    # parser.add_argument("--n_class", default=4, type=int)
    # parser.add_argument("--num_workers", default=8, type=int)
    # parser.add_argument("--dataset", default="luad", type=str)
    # parser.add_argument("--batch_size", default=16, type=int)
    
    # # Đường dẫn
    # parser.add_argument("--checkpoint_path", default='checkpoints/stage1_baseline_luad.pth', type=str)
    # parser.add_argument("--trainroot", default='LUAD-HistoSeg/training/img/', type=str)
    # parser.add_argument("--mask_save_path", default='results/luad_baseline_masks_dense/', type=str,
    #                     help="Thư mục để lưu các nhãn giả 'dày đặc'.")
    
    # args = parser.parse_args()
    
    # create_dense_masks(args)
    print(len(os.listdir('results/luad_baseline_masks_dense/')))