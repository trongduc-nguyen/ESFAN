# tool/OneShotDataset.py

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
from collections import defaultdict
from torchvision import transforms

class OneShotDataset(Dataset):
    def __init__(self, dataroot, pseudo_mask_dir, correct_list_file, n_class, transforms=None):
        """
        Dataset cho việc huấn luyện One-Shot Segmentation.
        ĐÃ CẬP NHẬT để sử dụng NHÃN GIẢ (pseudo-mask) thay vì ground truth mask.
        
        Args:
            dataroot (str): Đường dẫn đến thư mục training gốc (chứa 'img/').
            pseudo_mask_dir (str): Đường dẫn đến thư mục chứa các nhãn giả đã được tạo.
            correct_list_file (str): Đường dẫn đến file .txt chứa danh sách các ảnh.
            n_class (int): Số lớp foreground.
            transforms (callable, optional): Các phép biến đổi.
        """
        super().__init__()
        # Gán các thuộc tính từ tham số đầu vào
        self.img_dir = os.path.join(dataroot, 'img/')
        self.mask_dir = pseudo_mask_dir # Sử dụng thư mục nhãn giả
        self.transforms = transforms
        self.n_class = n_class

        # Các bước còn lại không đổi
        self.grouped_files = self._group_files_by_class_pattern(correct_list_file)
        self.samples = self._create_sample_list()

        print(f"Dataset được khởi tạo. Sử dụng nhãn giả từ: {self.mask_dir}")
        print(f"Tìm thấy {len(self.grouped_files)} nhóm lớp khác nhau.")
        print(f"Tổng số mẫu huấn luyện: {len(self.samples)}")

    def _group_files_by_class_pattern(self, list_file):
        """
        Đọc file .txt và nhóm các tên file theo label pattern.
        Ví dụ: {'[1,0,0,1]': ['file1.png', 'file2.png'], ...}
        """
        grouped = defaultdict(list)
        with open(list_file, 'r') as f:
            for line in f:
                filename = line.strip()
                if not filename:
                    continue
                
                try:
                    # Trích xuất pattern, ví dụ: '[1 0 0 1]'
                    pattern = filename.split('[')[-1].split(']')[0]
                    grouped[pattern].append(filename)
                except IndexError:
                    print(f"Cảnh báo: Không thể trích xuất pattern từ tên file: {filename}")
        
        # Lọc ra các nhóm chỉ có 1 ảnh, vì chúng không thể tạo cặp query
        filtered_grouped = {k: v for k, v in grouped.items() if len(v) > 1}
        return filtered_grouped

    def _create_sample_list(self):
        """
        Tạo một danh sách phẳng các mẫu để dễ dàng truy cập bằng index.
        """
        samples = []
        for pattern, filenames in self.grouped_files.items():
            for filename in filenames:
                samples.append((pattern, filename))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Trả về một cặp support-query và các mask tương ứng.
        """
        # 1. Lấy thông tin của ảnh support từ index
        support_pattern, support_filename = self.samples[index]
        
        # 2. Chọn ngẫu nhiên một ảnh query từ CÙNG NHÓM pattern
        possible_queries = self.grouped_files[support_pattern]
        # Đảm bảo query không phải là chính support
        query_filename = support_filename
        while query_filename == support_filename:
            query_filename = random.choice(possible_queries)
            
        # 3. Xây dựng đường dẫn đầy đủ
        support_img_path = os.path.join(self.img_dir, support_filename)
        support_mask_path = os.path.join(self.mask_dir, support_filename)
        
        query_img_path = os.path.join(self.img_dir, query_filename)
        query_mask_path = os.path.join(self.mask_dir, query_filename)
        
        # 4. Đọc ảnh và mask từ đường dẫn
        try:
            support_img = Image.open(support_img_path).convert('RGB')
            support_mask = Image.open(support_mask_path) # Mask đọc ở dạng gốc
            
            query_img = Image.open(query_img_path).convert('RGB')
            query_mask = Image.open(query_mask_path)
        except FileNotFoundError as e:
            print(f"Lỗi không tìm thấy file: {e}")
            # Trả về dữ liệu rỗng nếu có lỗi để DataLoader có thể bỏ qua
            return None 

        # 5. Áp dụng các phép biến đổi (data augmentation)
        # Quan trọng: Cùng một phép biến đổi phải được áp dụng cho cả ảnh và mask
        # và cặp support/query có thể có các biến đổi khác nhau
        sample_support = {'image': support_img, 'label': support_mask}
        sample_query = {'image': query_img, 'label': query_mask}
        
        if self.transforms:
            sample_support = self.transforms(sample_support)
            sample_query = self.transforms(sample_query)

        # 6. Chuẩn hóa mask: Chuyển sang dạng LongTensor và đảm bảo giá trị hợp lệ
        # Chúng ta cần mask của query để tính loss, và mask của support để tạo prototype
        support_mask_tensor = torch.from_numpy(np.array(sample_support['label'])).long()
        query_mask_tensor = torch.from_numpy(np.array(sample_query['label'])).long()
        
        return {
            'support_img': sample_support['image'],
            'support_mask': support_mask_tensor,
            'query_img': sample_query['image'],
            'query_mask': query_mask_tensor
        }

def collate_fn_one_shot(batch):
    """
    Hàm collate tùy chỉnh để xử lý trường hợp một sample có thể là None (do lỗi đọc file).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        # Trả về một dictionary rỗng nếu cả batch đều lỗi
        return {}
    
    # Sử dụng collate mặc định của PyTorch cho batch đã được lọc
    return torch.utils.data.dataloader.default_collate(batch)