# File: tool/infer_fun_joint.py (ĐÃ SỬA LỖI IMPORT VÀ LOGIC)

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import os
from tqdm import tqdm
from torchvision import transforms

# --- THAY ĐỔI IMPORT ---
# Import đúng Dataset mà chúng ta đã định nghĩa
from .simple_image_dataset import SimpleImageDataset

def infer_joint(model, dataroot, n_class, args):
    """
    Hàm inference cho JointModel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # --- Chuẩn bị DataLoader ---
    # Transform chỉ resize, Dataset sẽ chuyển kết quả thành numpy
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size))
    ])
    infer_dataset = SimpleImageDataset(dataroot=dataroot, transform=transform)







    
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)

    gt_list = []
    pred_list = []
    
    # to_tensor_norm = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    to_tensor_norm = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])

    # Vòng lặp này giờ sẽ nhận (img_names, np_images) một cách chính xác
    for img_names, np_images in tqdm(infer_data_loader, desc="Inferring with Joint Model"):
        
        # Chuyển numpy array sang tensor
        img_tensor = torch.stack([to_tensor_norm(img) for img in np_images]).to(device)

        with torch.no_grad():
            # Lấy feature map cuối cùng từ encoder
            _, _, features_before_pool = model.encoder(img_tensor)
            cam_weights = model.classifier_head.fc8.weight
            cam_raw = F.conv2d(features_before_pool, cam_weights)
            cam_raw = F.relu(cam_raw)
        
        for i in range(len(img_names)):
            img_name = img_names[i]
            
            gt_mask_path = os.path.join(dataroot, 'mask/', img_name)
            gt_mask = np.array(Image.open(gt_mask_path))
            
            present_classes = np.unique(gt_mask)
            label = torch.zeros(n_class)
            for c in present_classes:
                if c < n_class:
                    label[c] = 1.0
            
            cam_i = cam_raw[i:i+1]
            cam_upsampled = F.interpolate(cam_i, size=gt_mask.shape, mode='bilinear', align_corners=False)[0]
            
            cam_scores = cam_upsampled.cpu().numpy() * label.clone().view(n_class, 1, 1).numpy()
            pred_mask = np.argmax(cam_scores, axis=0)
        
            pred_mask[gt_mask == 4] = 4

            gt_list.append(gt_mask)
            pred_list.append(pred_mask)
            
    return gt_list, pred_list