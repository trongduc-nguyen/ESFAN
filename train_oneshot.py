# File: train_oneshot.py (CẬP NHẬT ĐỂ SỬ DỤNG TRUE_BG_MASK)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from oneshot_model import OneShotSegModel
from oneshot_dataset import OneShotPathologyDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Khởi tạo model và dataset...")
    model = OneShotSegModel(pretrained_path=args.pretrained_path).to(device)
    
    train_dataset = OneShotPathologyDataset(data_root=args.data_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Bắt đầu quá trình huấn luyện...")
    # Bắt đầu từ epoch được chỉ định (hữu ích khi resume)
    start_epoch = args.start_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss_epoch = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            support_image = batch['support_image'].to(device)
            support_mask = batch['support_mask'].to(device)
            query_image = batch['query_image'].to(device)
            query_label = batch['query_label'].to(device)
            # ***** THÊM DÒNG NÀY *****
            support_true_bg_mask = batch['support_true_bg_mask'].to(device)
            
            optimizer.zero_grad()
            
            # --- Forward Pass chính: Chỉ chạy encoder 1 LẦN ---
            # ***** THÊM THAM SỐ VÀO ĐÂY *****
            predicted_query_scores, support_features, query_features = model(
                support_image, support_mask, query_image, support_true_bg_mask
            )
            
            # --- Tính Loss 1 (chuẩn) ---
            loss_1 = criterion(predicted_query_scores, query_label)
            
            # --- Tính Loss 2 (Alignment Loss, tái sử dụng features) ---
            with torch.no_grad():
                predicted_query_mask = torch.argmax(predicted_query_scores, dim=1, keepdim=True).float()
            
            # ***** THÊM THAM SỐ VÀO ĐÂY (VỚI GIÁ TRỊ None) *****
            # Khi hoán đổi vai trò, chúng ta không có true_bg_mask cho query, nên truyền None
            predicted_support_scores = model.perform_classification(
                query_features, predicted_query_mask, support_features, support_true_bg_mask=None
            )
            
            support_label = support_mask.squeeze(1).long()
            loss_2 = criterion(predicted_support_scores, support_label)

            # --- Tổng hợp loss và Backward ---
            total_loss = loss_1 + args.lambda_val * loss_2
            total_loss.backward()
            optimizer.step()
            
            total_loss_epoch += total_loss.item()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", loss1=f"{loss_1.item():.4f}", loss2=f"{loss_2.item():.4f}")
            
        avg_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"oneshot_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Đã lưu checkpoint tại: {checkpoint_path}")

    print("Hoàn tất huấn luyện!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train One-Shot Segmentation Model")
    
    parser.add_argument('--data_root', type=str, default='LUAD-HistoSeg_OneShot/', help='Đường dẫn đến thư mục dữ liệu one-shot đã xử lý.')
    parser.add_argument('--pretrained_path', type=str, 
                        default='/home/25duc.nt3/ESFAN/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', 
                        help='Đường dẫn để load checkpoint và tiếp tục train, hoặc khởi tạo.')
    
    parser.add_argument('--save_dir', type=str, default='checkpoints_oneshot/', help='Thư mục để lưu các checkpoint.')
    parser.add_argument('--epochs', type=int, default=50, help='Tổng số epoch để huấn luyện.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch bắt đầu, dùng để resume training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Kích thước batch.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--lambda_val', type=float, default=0.3, help='Hệ số cho loss đối xứng (loss_2).')
    parser.add_argument('--num_workers', type=int, default=8, help='Số luồng cho DataLoader.')
    parser.add_argument('--save_interval', type=int, default=5, help='Lưu checkpoint sau mỗi số epoch này.')
    
    args = parser.parse_args()
    main(args)