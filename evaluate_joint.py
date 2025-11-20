# File: evaluate_joint.py

import argparse
import torch
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

# Import các thành phần cần thiết
from joint_model import JointModel
from tool import iouutils
from tool.simple_image_dataset import SimpleImageDataset
def main():
    parser = argparse.ArgumentParser(description="Evaluate CAM from Jointly Trained Model")
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--checkpoint_path", default='checkpoints_joint/joint_model_epoch_25.pth', type=str)
    parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    # Thêm action để tương thích hoàn toàn, mặc dù chúng ta chỉ dùng 'evaluate'
    parser.add_argument("--action", default='evaluate', choices=['evaluate'])
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 1. Khởi tạo và Load JointModel ---
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model = JointModel(n_class=args.n_class)
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    except RuntimeError:
        print("Phát hiện checkpoint từ DataParallel, đang sửa lại keys...")
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    print("JointModel loaded successfully.")

    # --- 2. CHUẨN BỊ DATALOADER (FIX CHUẨN) ---
    # Transform giờ bao gồm cả ToTensor
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor() # Thêm ToTensor ở đây
    ])
    infer_dataset = SimpleImageDataset(dataroot=args.testroot, transform=transform)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)

    # --- 3. VÒNG LẶP ĐÁNH GIÁ (FIX CHUẨN) ---
    gt_list = []
    cam_pred_list = []
    
    # Define Normalize transform riêng để áp dụng sau
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for img_names, imgs in tqdm(infer_data_loader, desc="Evaluating CAM from Joint Model"):
        
        # DataLoader đã trả về một batch tensor
        img_tensor = imgs.to(device)
        
        # Normalize batch tensor
        img_tensor = normalize(img_tensor)

        # Lấy label từ GT mask (logic này đã đúng từ lần trước)
        labels_list = []
        for name in img_names:
            gt_mask_path = os.path.join(args.testroot, 'mask/', name)
            gt_mask = np.array(Image.open(gt_mask_path))
            present_classes = np.unique(gt_mask)
            label_vec = torch.zeros(args.n_class)
            for c in present_classes:
                if c < args.n_class:
                    label_vec[c] = 1.0
            labels_list.append(label_vec)
        labels = torch.stack(labels_list)

        with torch.no_grad():
            _, _, features_before_pool = model.encoder(img_tensor)
            cam_weights = model.classifier_head.fc8.weight
            cam_raw = F.conv2d(features_before_pool, cam_weights)
            cam_raw = F.relu(cam_raw)
        
        for i in range(len(img_names)):
            img_name = img_names[i]
            label = labels[i]
            gt_mask_path = os.path.join(args.testroot, 'mask/', img_name)
            gt_mask = np.array(Image.open(gt_mask_path))
            
            cam_i = cam_raw[i:i+1]
            cam_upsampled = F.interpolate(cam_i, size=gt_mask.shape, mode='bilinear', align_corners=False)[0]
            cam_scores = cam_upsampled.cpu().numpy() * label.clone().view(args.n_class, 1, 1).numpy()
            cam_mask = np.argmax(cam_scores, axis=0)
        
            cam_mask[gt_mask == 4] = 4
            gt_list.append(gt_mask)
            cam_pred_list.append(cam_mask)

    # --- 4. TÍNH TOÁN VÀ IN KẾT QUẢ ---
    # (Phần này không thay đổi)
    print("\n--- KẾT QUẢ ĐÁNH GIÁ (CAM TỪ JOINT MODEL) ---")
    results = iouutils.scores(gt_list, cam_pred_list, n_class=args.n_class)
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()