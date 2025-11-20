## This code is a example for using the image level annotation of training patches.

import os
import torch

file = 'TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png'

fname = file[:-4]
print(fname)

label_str = fname.split(']')[0].split('[')[-1]
print(label_str)

torch_label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])
print(torch_label)