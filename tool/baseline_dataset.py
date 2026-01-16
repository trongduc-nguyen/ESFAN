import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BaselineInferDataset(Dataset):
    def __init__(self, dataroot, n_class, transform=None):
        """
        Lớp Dataset đa năng:
        - Tự động phát hiện cấu trúc thư mục (Train hay Test).
        - Nếu là Test (có folder 'img/'): Đọc ảnh từ 'img/' và nhãn từ 'mask/'.
        - Nếu là Train (không có folder 'img/'): Đọc ảnh trực tiếp và parse nhãn từ tên file.
        """
        super().__init__()
        self.dataroot = dataroot
        self.n_class = n_class
        self.transform = transform
        print("dataroot", dataroot)
        # Kiểm tra cấu trúc thư mục để xác định chế độ
        if os.path.exists(os.path.join(dataroot, 'img')):
            self.mode = 'test_structure'
            self.img_dir = os.path.join(dataroot, 'img/')
            self.mask_dir = os.path.join(dataroot, 'mask/')
            print(f"-> Phát hiện cấu trúc TEST (có thư mục img/). Đọc từ: {self.img_dir}")
        else:
            self.mode = 'train_structure'
            self.img_dir = dataroot
            self.mask_dir = None
            print(f"-> Phát hiện cấu trúc TRAIN (ảnh nằm ở root). Đọc từ: {self.img_dir}")

        # Lấy danh sách file ảnh
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.png')]
        
        if not self.filenames:
            raise FileNotFoundError(f"Không tìm thấy ảnh .png nào trong: {self.img_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_name = self.filenames[index]
        img_path = os.path.join(self.img_dir, img_name + '.png')
        
        # Đọc ảnh
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path}: {e}")
            # Trả về ảnh đen nếu lỗi
            img = Image.new('RGB', (256, 256))

        if self.transform is not None:
            img = self.transform(img)
            
        # --- XỬ LÝ NHÃN (IMAGE-LEVEL LABEL) ---
        label = torch.zeros(self.n_class)
        
        if self.mode == 'test_structure':
            # CASE 1: Cấu trúc Test (Lấy label từ file Mask GT)
            mask_path = os.path.join(self.mask_dir, img_name + '.png')
            if os.path.exists(mask_path):
                try:
                    gt_mask = np.array(Image.open(mask_path))
                    present_classes = np.unique(gt_mask)
                    for cls_idx in present_classes:
                        # 0-3 là foreground, 4 là background
                        if 0 <= cls_idx < self.n_class:
                            label[cls_idx] = 1.0
                except:
                    pass
                    
        elif self.mode == 'train_structure':
            # CASE 2: Cấu trúc Train (Parse label từ tên file)
            # Hỗ trợ cả 2 định dạng:
            # - LUAD: "...-[1 0 0 1].png" (có dấu cách)
            # - BCSS: "...-[0001].png" (không dấu cách)
            try:
                if '[' in img_name and ']' in img_name:
                    label_str = img_name.split(']')[0].split('[')[-1]
                    # Xóa khoảng trắng để chuẩn hóa
                    label_str = label_str.replace(' ', '')
                    
                    # Parse từng ký tự thành số
                    # Ví dụ: "1001" -> [1, 0, 0, 1]
                    indices = [int(c) for c in label_str]
                    
                    if len(indices) == self.n_class:
                        label = torch.Tensor(indices)
                    else:
                        print(f"Warning: Độ dài nhãn không khớp n_class ({self.n_class}) cho file: {img_name}")
            except Exception as e:
                print(f"Warning: Lỗi parse nhãn file {img_name}: {e}")

        return img_name, img, label