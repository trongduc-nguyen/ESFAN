# File: train_joint.py (PHIÊN BẢN HOÀN CHỈNH - HUẤN LUYỆN ĐỒNG THỜI ĐA INSTANCE)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from skimage import morphology
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Import các thành phần cần thiết
from joint_model import JointModel
from tool.GenDataset import Stage1_TrainDataset
from tool import torchutils, pyutils
from tool import custom_transforms_oneshot as tr # Import file augmentation

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 1. Khởi tạo Model ---
    # Sửa đổi JointModel để forward_classification có thể trả về feature maps
    model = JointModel(n_class=args.n_class, pretrained_path=args.weights).to(device)

    # --- 2. Khởi tạo Dataloaders và Transforms ---
    print("Khởi tạo DataLoaders và Transforms...")
    
    # DataLoader cho classification sẽ trả về ảnh PIL để dễ dàng augment on-the-fly
    transform_cls_base = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    cls_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_cls_base, dataset=args.dataset)
    cls_loader = DataLoader(cls_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # Các hàm transform sẽ được dùng trong vòng lặp
    augment_transform = tr.RandomAffine(
        rotation_range=30,
        translation_range=(0.2 * args.input_size, 0.2 * args.input_size),
        zoom_range=(0.8, 1.2)
    )
    to_tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. Khởi tạo Optimizer và Loss ---
    max_step = (len(cls_dataset) // args.batch_size) * args.epochs
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    criterion_cls = nn.MultiLabelSoftMarginLoss()
    criterion_oneshot = nn.CrossEntropyLoss()
    
    print("Bắt đầu quá trình huấn luyện đồng thời...")
    start_epoch = args.start_epoch

    temp_lambda_oneshot = args.lambda_oneshot
    temp_relative_conf_threshold = args.relative_conf_threshold
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(cls_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_data in progress_bar:
            optimizer.zero_grad()
            
            _, pil_images, cls_labels = batch_data
            cls_labels = cls_labels.to(device)

            # Chuyển đổi PIL images sang tensor cho nhánh classification
            np_images = [np.array(img) for img in pil_images]
            cls_images_tensor = torch.stack([to_tensor_transform(Image.fromarray(img)) for img in np_images]).to(device)

            # --- Nhánh 1: Loss Classification & Tạo CAMs ---
            # x1, x2, x, _, all_features_cls = model.classifier_head(cls_images_tensor, return_features=True) # Yêu cầu model trả về features
            x1, x2, x, _, _, conv6 = model.classifier_head(cls_images_tensor, return_features=True)
            loss_cls = 0.2 * criterion_cls(x1, cls_labels) + 0.3 * criterion_cls(x2, cls_labels) + 0.5 * criterion_cls(x, cls_labels)
            
            with torch.no_grad():
                cam_weights = model.classifier_head.fc8.weight
                cams = F.conv2d(conv6.detach(), cam_weights)
                cams = F.relu(cams)
                cams_upsampled = F.interpolate(cams, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False).cpu().numpy()

            # --- Nhánh 2: Loss One-Shot (tính trên cùng batch) ---
            loss_oneshot_accumulator = []
            
            for i in range(args.batch_size):
                try:

                    image_np = np_images[i]
                    image_label = cls_labels[i].cpu()
                    cam_scores = cams_upsampled[i] * image_label.clone().view(args.n_class, 1, 1).numpy()
                    
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                    true_bg_mask = morphology.remove_small_objects(binary==255, min_size=50, connectivity=1)

                    present_classes_indices = torch.where(image_label == 1)[0].numpy()
                    if len(present_classes_indices) > 1:
                        for class_idx in present_classes_indices:
                            try:

                                class_cam = cam_scores[class_idx]
                                if class_cam.max() <= 0: continue

                                normalized_cam = (class_cam - class_cam.min()) / (class_cam.max() - class_cam.min())
                                support_mask_np = (normalized_cam > temp_relative_conf_threshold)
                                support_mask_np[true_bg_mask] = False
                                support_mask_np = morphology.remove_small_objects(support_mask_np, min_size=100, connectivity=1).astype(np.float32)

                                if support_mask_np.sum() == 0: continue
                                
                                support_image_np = image_np.copy()
                                query_sample = {'image': support_image_np, 'mask': support_mask_np}
                                augmented = augment_transform(query_sample)
                                query_image_np = augmented['image']
                                query_mask_np = augmented['mask']
                                
                                support_image = to_tensor_transform(Image.fromarray(support_image_np)).unsqueeze(0).to(device)
                                support_mask = torch.from_numpy(support_mask_np).unsqueeze(0).unsqueeze(0).to(device)
                                support_true_bg_mask = torch.from_numpy(true_bg_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                                query_image = to_tensor_transform(Image.fromarray(query_image_np)).unsqueeze(0).to(device)
                                query_label = torch.from_numpy(query_mask_np).long().unsqueeze(0).to(device)

                                predicted_q_scores, sup_feat, qry_feat = model.forward_oneshot(
                                    support_image, support_mask, query_image, support_true_bg_mask
                                )
                                
                                loss_os_1 = criterion_oneshot(predicted_q_scores, query_label)
                                
                                with torch.no_grad():
                                    pred_q_mask = torch.argmax(predicted_q_scores, dim=1, keepdim=True).float()
                                
                                pred_s_scores = model.perform_oneshot_classification(
                                    qry_feat, pred_q_mask, sup_feat, support_true_bg_mask=None
                                )
                                
                                loss_os_2 = criterion_oneshot(pred_s_scores, support_mask.squeeze(1).long())
                                loss_instance = loss_os_1 + args.lambda_align * loss_os_2
                                loss_oneshot_accumulator.append(loss_instance)
                            except Exception as e:
                                print(f"Warning: Skipping one instance due to error: {e}")

                            
                except Exception as e:
                    print(f"Lỗi khi xử lý mẫu {i} trong batch: {e}")
            # --- Tổng hợp Loss và Backward ---
            if loss_oneshot_accumulator:
                loss_oneshot = torch.mean(torch.stack(loss_oneshot_accumulator))
                total_loss = loss_cls + temp_lambda_oneshot * loss_oneshot
            else:
                total_loss = loss_cls
                loss_oneshot = torch.tensor(0.0) # Để log

            total_loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss_total=f"{total_loss.item():.4f}", 
                                     loss_cls=f"{loss_cls.item():.4f}", 
                                     loss_os=f"{loss_oneshot.item():.4f}")

        # --- Lưu checkpoint ---
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"joint_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Đã lưu checkpoint tại: {checkpoint_path}")
        if (epoch + 1) % 10 == 0:
            temp_lambda_oneshot = temp_lambda_oneshot * 1.2
            temp_relative_conf_threshold = temp_relative_conf_threshold - 0.1
            
    print("Hoàn tất huấn luyện đồng thời!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Jointly (On-the-fly Multi-Instance)")
        # Tham số chung
    # parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--save_dir", default='checkpoints_joint/', type=str)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--dataset", default='luad', type=str)

    # ... (Các tham số lr, wt_dec, n_class, save_dir, save_interval, dataset giữ nguyên)
    parser.add_argument("--batch_size", default=8, type=int, help="Lưu ý: batch size này sẽ tạo ra nhiều tác vụ one-shot, nên chọn nhỏ hơn.")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=224)

    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='LUAD-HistoSeg/training/img', type=str)
    
    parser.add_argument("--lambda_oneshot", type=float, default=0.05, help="Hệ số cho one-shot loss.")
    parser.add_argument("--lambda_align", type=float, default=0.3, help="Hệ số cho one-shot alignment loss.")
    parser.add_argument('--relative_conf_threshold', type=float, default=0.3, help='Ngưỡng tin cậy tương đối để tạo support mask on-the-fly.')

    args = parser.parse_args()
    main(args)