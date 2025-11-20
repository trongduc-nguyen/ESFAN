# File: tool/alpmodule.py (CẬP NHẬT ĐỂ LỌC PROTOTYPE TỪ BACKGROUND THẬT)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode = 'bilinear'):
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [int(ft_l // grid_l) for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)

    def safe_norm(self, x, p=2, dim=1, eps=1e-4):
        x_norm = torch.norm(x, p=p, dim=dim)
        x_norm = torch.max(x_norm, torch.ones_like(x_norm, device=x.device) * eps)
        x = x.div(x_norm.unsqueeze(1).expand_as(x))
        return x
        
    def forward(self, qry_fts, sup_fts, sup_mask, true_bg_mask=None, mode='gridconv+', thresh=0.95):
        """
        Phiên bản cập nhật để lọc bỏ prototypes từ vùng background thật.
        Args:
            qry_fts (torch.Tensor): [B, C, Hf, Wf]
            sup_fts (torch.Tensor): [B, C, Hf, Wf]
            sup_mask (torch.Tensor): [B, 1, Hf, Wf]
            true_bg_mask (torch.Tensor, optional): Mask của vùng background thật, [B, 1, Hf, Wf].
            mode (str): 'mask', 'gridconv', hoặc 'gridconv+'
        """
        batch_size, n_channels, h_fts, w_fts = qry_fts.shape

        if mode == 'mask':  # Chỉ global prototype, không bị ảnh hưởng bởi true_bg_mask
            proto = torch.sum(sup_fts * sup_mask, dim=(-1, -2)) / (sup_mask.sum(dim=(-1, -2)) + 1e-5)
            pred_mask = F.cosine_similarity(qry_fts, proto.unsqueeze(-1).unsqueeze(-1), dim=1, eps=1e-4)
            return pred_mask.unsqueeze(1)

        elif mode == 'gridconv': # Chỉ local prototypes
            # Tạo các lưới đặc trưng, mask, và background mask
            local_sup_fts = self.avg_pool_op(sup_fts)
            local_sup_mask = self.avg_pool_op(sup_mask)
            if true_bg_mask is not None:
                local_true_bg_mask = self.avg_pool_op(true_bg_mask)
            
            # Reshape để dễ lọc
            local_sup_fts = local_sup_fts.view(batch_size, n_channels, -1).permute(0, 2, 1)
            local_sup_mask = local_sup_mask.view(batch_size, 1, -1).permute(0, 2, 1)
            if true_bg_mask is not None:
                local_true_bg_mask = local_true_bg_mask.view(batch_size, 1, -1).permute(0, 2, 1)

            protos_list = []
            for i in range(batch_size):
                # Mask cho các prototype có đủ tín hiệu từ support mask
                valid_signal_mask = local_sup_mask[i].squeeze() > thresh

                # ***** LOGIC LỌC MỚI *****
                if true_bg_mask is not None:
                    # Mask cho các prototype KHÔNG nằm trong vùng background thật
                    not_true_bg_mask = local_true_bg_mask[i].squeeze() < 0.5
                    # Mask cuối cùng: phải có tín hiệu VÀ không phải background thật
                    final_valid_mask = valid_signal_mask & not_true_bg_mask
                else:
                    final_valid_mask = valid_signal_mask
                # *************************
                
                valid_protos = local_sup_fts[i][final_valid_mask]
                protos_list.append(valid_protos)
            
            # Phần tính toán so khớp còn lại giữ nguyên
            scores_list = []
            for i in range(batch_size):
                if len(protos_list[i]) == 0:
                    scores_list.append(torch.zeros_like(qry_fts[i:i+1, 0:1, ...]))
                    continue
                protos_norm = self.safe_norm(protos_list[i])
                qry_norm = self.safe_norm(qry_fts[i:i+1])
                dists = F.conv2d(qry_norm, protos_norm.unsqueeze(-1).unsqueeze(-1))
                pred_grid = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
                scores_list.append(pred_grid)
            return torch.cat(scores_list, dim=0)

        elif mode == 'gridconv+': # Cả local và global
            # Logic tương tự như 'gridconv'
            local_sup_fts = self.avg_pool_op(sup_fts)
            local_sup_mask = self.avg_pool_op(sup_mask)
            if true_bg_mask is not None:
                local_true_bg_mask = self.avg_pool_op(true_bg_mask)
            
            local_sup_fts = local_sup_fts.view(batch_size, n_channels, -1).permute(0, 2, 1)
            local_sup_mask = local_sup_mask.view(batch_size, 1, -1).permute(0, 2, 1)
            if true_bg_mask is not None:
                local_true_bg_mask = local_true_bg_mask.view(batch_size, 1, -1).permute(0, 2, 1)

            global_proto = torch.sum(sup_fts * sup_mask, dim=(-1, -2)) / (sup_mask.sum(dim=(-1, -2)) + 1e-5)
            
            scores_list = []
            for i in range(batch_size):
                valid_signal_mask = local_sup_mask[i].squeeze() > thresh
                
                # ***** LOGIC LỌC MỚI *****
                if true_bg_mask is not None:
                    not_true_bg_mask = local_true_bg_mask[i].squeeze() < 0.5
                    final_valid_mask = valid_signal_mask & not_true_bg_mask
                else:
                    final_valid_mask = valid_signal_mask
                # *************************
                
                valid_local_protos = local_sup_fts[i][final_valid_mask]
                
                # Global prototype không cần lọc, nên luôn được thêm vào
                all_protos = torch.cat([valid_local_protos, global_proto[i:i+1]], dim=0)

                if len(all_protos) == 0:
                    scores_list.append(torch.zeros_like(qry_fts[i:i+1, 0:1, ...]))
                    continue
                    
                protos_norm = self.safe_norm(all_protos)
                qry_norm = self.safe_norm(qry_fts[i:i+1])
                dists = F.conv2d(qry_norm, protos_norm.unsqueeze(-1).unsqueeze(-1))
                pred_grid = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
                scores_list.append(pred_grid)
            return torch.cat(scores_list, dim=0)
        else:
            raise NotImplementedError