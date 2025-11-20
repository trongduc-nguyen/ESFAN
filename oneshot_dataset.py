# File: oneshot_dataset.py (CẬP NHẬT LẦN CUỐI ĐỂ TRẢ VỀ TRUE_BG_MASK)

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from skimage import morphology
from tool import custom_transforms_oneshot as tr 

class OneShotPathologyDataset(Dataset):
    """
    Dataset cho multi-task. 
    Mỗi sample trả về ảnh gốc, instance mask đa nhãn, và bg mask.
    """
    def __init__(self, data_root, input_size=(224, 224)):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'images')
        self.mask_dir = os.path.join(data_root, 'instance_masks')
        self.image_files = sorted(os.listdir(self.image_dir))
        self.input_size = input_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image_pil = Image.open(img_path).convert("RGB").resize(self.input_size, Image.BILINEAR)
        instance_mask_pil = Image.open(mask_path).resize(self.input_size, Image.NEAREST)
        
        # Chuyển sang numpy
        image_np = np.array(image_pil)
        instance_mask_np = np.array(instance_mask_pil)

        instance_ids = np.unique(instance_mask_np)
        instance_ids = instance_ids[instance_ids != 0]

        if len(instance_ids) == 0:
            return self.__getitem__((index + 1) % len(self))

        # ***** THÊM LOGIC TÍNH TRUE_BG_MASK Ở ĐÂY *****
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary_mask = (binary == 255)
        true_bg_mask_np = morphology.remove_small_objects(binary_mask, min_size=50, connectivity=1).astype(np.float32)

        return {
            'image_pil': image_pil,
            'image_np': image_np, # Trả về cả dạng numpy để dùng lại
            'instance_mask_np': instance_mask_np,
            'true_bg_mask_np': true_bg_mask_np, # TRẢ VỀ KEY CÒN THIẾU
        }

def custom_collate_fn(batch):
    """
    Collate function tùy chỉnh.
    """
    return batch