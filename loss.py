import numpy as np
import torch
import torch.nn as nn


class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta, reg):
        super(VoxelLoss, self).__init__()
        self.eps = 1e-6
        self.alpha = alpha
        self.beta = beta
        self.reg = reg
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):
        # regression map, possibility score map: h/2, w/2, 14,
        p_pos = torch.sigmoid(psm.permute(0, 2, 3, 1))
        d_peo = pos_equal_one.dim()
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(d_peo).expand(-1, -1, -1, -1, 7)
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), rm.size(1), rm.size(2), -1, 7)
        targets = targets.view(targets.size(0), targets.size(1), targets.size(2), -1, 7)

        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg
        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = self.reg * reg_loss / (pos_equal_one.sum() + self.eps)

        cls_pos_loss = (-pos_equal_one * torch.log(p_pos + self.eps)).sum()/(pos_equal_one.sum()+self.eps)
        cls_neg_loss = (-neg_equal_one * torch.log(1 - p_pos + self.eps)).sum()/(neg_equal_one.sum()+self.eps)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

        return conf_loss, reg_loss
