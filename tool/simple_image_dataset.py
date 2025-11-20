# File: tool/simple_image_dataset.py (FIX CHUẨN)

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SimpleImageDataset(Dataset):
    def __init__(self, dataroot, transform=None):
        self.img_dir = os.path.join(dataroot, 'img')
        if not os.path.isdir(self.img_dir):
            raise ValueError(f"Thư mục ảnh không tồn tại: {self.img_dir}")
        self.filenames = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_name = self.filenames[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # transform giờ sẽ bao gồm cả ToTensor
        if self.transform:
            img = self.transform(img)

        # Trả về (tên file, tensor ảnh)
        return img_name, img