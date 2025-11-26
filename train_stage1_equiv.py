import os
import numpy as np
import argparse
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset

cudnn.enabled = True
# Lưu ý: Set device phù hợp với server của bạn
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def train_phase(args):
    # Load Model
    model = getattr(importlib.import_module(args.network), 'Net')(n_class=args.n_class)
    print(vars(args))
    
    # Data Augmentation & Loading
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])
    
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    
    # Optimizer Settings
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    # Load Weights
    if args.weights and args.weights.endswith('.params'):
        import network.resnet38d 
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        model.backbone.load_state_dict(weights_dict, strict=False)
        print("Pre-trained weights loaded from .params file.")
    elif args.weights and args.weights.endswith('.pth'):
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
        print("Pre-trained weights loaded from .pth file.")
    else:
        print('Training from scratch (random init).')
        
    model = model.cuda()
    
    # Meters
    avg_meter = pyutils.AverageMeter('loss_cls', 'loss_er', 'loss_total')
    timer = pyutils.Timer("Session started: ")
    
    for ep in range(args.max_epoches):
        model.train()
        for iter, (filename, data, label) in enumerate(train_data_loader):
            img = data.cuda()
            label = label.cuda(non_blocking=True)
            
            # ------------------------------------------------------------------
            # NOVELTY: Equivariant Regularization (Scale Consistency)
            # ------------------------------------------------------------------
            
            # 1. Forward Branch Gốc (Original Image)
            # Cần lấy conv6 để tự tính CAM
            x1, x2, x, feature, y, conv6 = model(img, return_features=True)
            
            # --- Classification Loss (Baseline) ---
            loss1 = F.multilabel_soft_margin_loss(x1, label)
            loss2 = F.multilabel_soft_margin_loss(x2, label)
            loss3 = F.multilabel_soft_margin_loss(x, label)
            loss_cls = 0.2*loss1 + 0.3*loss2 + 0.5*loss3
            
            # 2. Forward Branch Biến đổi (Scaled Image)
            # Tạo scale ngẫu nhiên từ 0.7 đến 1.3
            scale_factor = np.random.uniform(0.7, 1.3)
            img_scaled = F.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            
            # Forward ảnh scaled qua mạng
            # Ta chỉ cần conv6_s để tính CAM, không cần tính loss classification cho nhánh này (để tiết kiệm hoặc tập trung regularization)
            _, _, _, _, _, conv6_s = model(img_scaled, return_features=True)
            
            # 3. Tính CAM thủ công
            # CAM = Conv6 * Weight_FC8
            fc8_weights = model.fc8.weight # [n_class, 4096, 1, 1]
            
            cam_orig = F.conv2d(conv6, fc8_weights)   # CAM của ảnh gốc
            cam_scaled = F.conv2d(conv6_s, fc8_weights) # CAM của ảnh scaled
            
            # 4. Alignment & Consistency Loss
            # Để so sánh, ta phải đưa CAM gốc về cùng kích thước với CAM scaled
            # (Hoặc ngược lại. Resize CAM gốc xuống CAM scaled thường ổn định hơn)
            target_h, target_w = cam_scaled.shape[2], cam_scaled.shape[3]
            cam_orig_aligned = F.interpolate(cam_orig, size=(target_h, target_w), mode='bilinear', align_corners=True)
            
            # Áp dụng ReLU để chỉ so sánh phần kích hoạt dương (Feature quan trọng)
            cam_orig_aligned = F.relu(cam_orig_aligned)
            cam_scaled = F.relu(cam_scaled)
            
            # Equivariant Loss: L1 Loss hoặc MSE Loss
            # Chỉ tính loss trên các class có trong nhãn (Foreground) để tránh nhiễu background
            # Hoặc tính trên toàn bộ để ép background consistency. Ở đây ta dùng toàn bộ cho đơn giản và mạnh.
            loss_er = torch.mean(torch.abs(cam_orig_aligned - cam_scaled))
            
            # 5. Total Loss
            loss_total = loss_cls + loss_er
            
            # ------------------------------------------------------------------
            
            avg_meter.add({
                'loss_cls': loss_cls.item(), 
                'loss_er': loss_er.item(), 
                'loss_total': loss_total.item()
            })
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            if (optimizer.global_step) % 100 == 0 and (optimizer.global_step) != 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                      'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                      'Loss_Tot:%.4f' % (avg_meter.get('loss_total')),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)

        # Save checkpoint
        if ep % 3 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_equiv_' + args.dataset + '_epoch_' + str(ep) + '.pth'))

    # Save final
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_equiv_' + args.dataset + '.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Giảm batch size nếu bị OOM do forward 2 lần")
    parser.add_argument("--max_epoches", default=25, type=int)
    parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str) 
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    
    # Dataset config
    parser.add_argument("--trainroot", default='LUAD-HistoSeg/training', type=str)
    parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str)
    parser.add_argument("--dataset", default='luad', type=str, choices=['luad', 'bcss'])
    
    parser.add_argument("--save_folder", default='checkpoints', type=str)
    
    # Novelty config
    parser.add_argument("--er_alpha", default=0.5, type=float, help="Trọng số cho Equivariant Regularization Loss")

    args = parser.parse_args()
    
    # Tạo folder save nếu chưa có
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        
    train_phase(args)