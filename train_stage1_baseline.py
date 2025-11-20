# train_stage1_baseline.py

import os
import numpy as np
import argparse
import importlib
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
# from tool.infer_fun import infer, get_mask
cudnn.enabled = True
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
    # Thay vì 'Net' từ resnet38_cls, chúng ta dùng 'Net' từ resnet38_cls_baseline
    model = getattr(importlib.import_module(args.network), 'Net')(n_class=args.n_class)
    print(vars(args))
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.ToTensor()])
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot,transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    # Logic load weights vẫn giữ nguyên để có thể dùng pre-trained weights
    if args.weights and args.weights.endswith('.params'):
        import network.resnet38d 
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        # Load vào backbone bên trong model của chúng ta
        model.backbone.load_state_dict(weights_dict, strict=False)
        print("Pre-trained weights loaded from .params file.")
    elif args.weights and args.weights.endswith('.pth'):
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
        print("Pre-trained weights loaded from .pth file.")
    else:
        print('Training from scratch (random init).')
        
    model = model.cuda()
    avg_meter = pyutils.AverageMeter('loss1','loss2','loss3','loss','avg_ep_EM','avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):
        model.train()
        for iter, (filename, data, label) in enumerate(train_data_loader):
            img = data.cuda()
            label = label.cuda(non_blocking=True)
            
            x1, x2, x, feature, y = model(img)
            
            prob = torch.sigmoid(y).cpu().data.numpy() # Dùng sigmoid cho multilabel
            gt = label.cpu().data.numpy()
            
            # Tính accuracy... (giữ nguyên logic)
            # ...
            
            loss1 = F.multilabel_soft_margin_loss(x1, label)
            loss2 = F.multilabel_soft_margin_loss(x2, label)
            loss3 = F.multilabel_soft_margin_loss(x, label)
            loss = 0.2*loss1 + 0.3*loss2 + 0.5*loss3 # Deep supervision
            
            avg_meter.add({'loss1':loss1.item(),'loss2': loss2.item(),'loss3': loss3.item(),'loss':loss.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (optimizer.global_step)%100 == 0 and (optimizer.global_step)!=0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)
                      
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_baseline_'+args.dataset+'.pth'))

# Các hàm test và gene_mask cần được cập nhật để dùng Net_CAM
# ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    # --- THAY ĐỔI QUAN TRỌNG ---
    parser.add_argument("--network", default="network.resnet38_cls_baseline", type=str) 
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    # Cần cung cấp file weights pre-trained của ResNet-38
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='LUAD-HistoSeg/training/', type=str)
    parser.add_argument("--testroot", default='LUAD-HistoSeg/test/', type=str)
    parser.add_argument("--save_folder", default='checkpoints', type=str)
    parser.add_argument("--dataset", default='luad', type=str)
    args = parser.parse_args()
    
    train_phase(args)
    # Tạm thời comment 2 phase sau, chúng ta sẽ làm sau khi train xong
    # test_phase(args) 
    # gene_mask(args)