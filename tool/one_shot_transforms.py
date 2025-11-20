# tool/one_shot_transforms.py

import torch
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
from torchvision import transforms
class Compose(object):
    """Gộp nhiều phép biến đổi lại với nhau."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Chuyển ảnh PIL và mask numpy thành Tensor."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # Chuyển ảnh PIL sang Tensor, tự động chuẩn hóa về [0, 1]
        image = F.to_tensor(image)
        # Chuyển mask (có thể là PIL hoặc numpy) thành LongTensor
        label = torch.from_numpy(np.array(label)).long()
        return {'image': image, 'label': label}

class Normalize(object):
    """Chuẩn hóa Tensor ảnh với mean và std cho trước."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, mean=self.mean, std=self.std)
        return {'image': image, 'label': label}

class RandomHorizontalFlip(object):
    """Lật ngang ngẫu nhiên ảnh và mask."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = F.hflip(image)
            label = F.hflip(label)
        return {'image': image, 'label': label}
        
class RandomVerticalFlip(object):
    """Lật dọc ngẫu nhiên ảnh và mask."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = F.vflip(image)
            label = F.vflip(label)
        return {'image': image, 'label': label}

class RandomRotation(object):
    """Xoay ngẫu nhiên ảnh và mask."""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = random.uniform(-self.degrees, self.degrees)
        # fill=4 vì lớp 4 là background, khi xoay các pixel trống sẽ được điền giá trị này
        image = F.rotate(image, angle)
        label = F.rotate(label, angle, fill=4)
        return {'image': image, 'label': label}

class RandomResizedCrop(object):
    """Cắt và thay đổi kích thước ngẫu nhiên."""
    def __init__(self, size, scale=(0.5, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Lấy các tham số cắt ngẫu nhiên
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=(0.75, 1.33))
        
        # Áp dụng cùng phép cắt cho cả ảnh và mask
        image = F.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
        # Dùng NEAREST cho mask để không tạo ra các giá trị pixel mới
        label = F.resized_crop(label, i, j, h, w, self.size, Image.NEAREST)
        
        return {'image': image, 'label': label}

class GaussianBlur(object):
    """Làm mờ Gaussian ngẫu nhiên."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return {'image': image, 'label': label}