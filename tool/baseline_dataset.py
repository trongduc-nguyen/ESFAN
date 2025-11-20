# tool/baseline_dataset.py
# File này chứa các lớp Dataset tùy chỉnh cho pipeline baseline của chúng ta,
# không làm ảnh hưởng đến các file gốc của tác giả.

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BaselineInferDataset(Dataset):
    def __init__(self, dataroot, n_class, transform=None):
        """
        Lớp Dataset tùy chỉnh cho việc đánh giá và tạo pseudo-mask.
        Nó đọc ảnh từ thư mục 'img/' và tự động suy ra image-level label
        bằng cách đọc file mask tương ứng từ thư mục 'mask/'.
        
        Args:
            dataroot (str): Đường dẫn đến thư mục gốc của tập dữ liệu
                            (vd: 'LUAD-HistoSeg/test/' hoặc 'LUAD-HistoSeg/training/').
                            Thư mục này phải chứa các thư mục con 'img/' và 'mask/'.
            n_class (int): Số lượng lớp foreground (ví dụ: 4).
            transform (callable, optional): Các phép biến đổi áp dụng cho ảnh.
        """
        super().__init__()
        self.dataroot = dataroot
        self.img_dir = os.path.join(dataroot, 'img/')
        self.mask_dir = os.path.join(dataroot, 'mask/')
        self.n_class = n_class
        self.transform = transform
        
        # Lấy danh sách tên file (không có đuôi) từ thư mục img/
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.png')]
        if not self.filenames:
            raise FileNotFoundError(f"Không tìm thấy ảnh .png nào trong thư mục: {self.img_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Lấy tên file
        img_name = self.filenames[index]
        
        # Xây dựng đường dẫn
        img_path = os.path.join(self.img_dir, img_name + '.png')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')
        
        # Đọc ảnh
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        # Đọc mask và tạo image-level label
        if not os.path.exists(mask_path):
            # Nếu không tìm thấy mask, trả về label rỗng (trường hợp này hiếm)
            label = torch.zeros(self.n_class)
        else:
            gt_mask = np.array(Image.open(mask_path))
            present_classes = np.unique(gt_mask)
            
            # Tạo image-level label (tensor n_class chiều)
            label = torch.zeros(self.n_class)
            for cls_idx in present_classes:
                # Quy ước đã xác định: 0-3 là foreground, 4 là background
                if 0 <= cls_idx < self.n_class:
                    label[cls_idx] = 1.0
        
        # Trả về 3 giá trị cần thiết
        return img_name, img, label