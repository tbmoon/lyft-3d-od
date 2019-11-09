import numpy as np
import torch
import torch.nn as nn
from config import config as cfg


class VoxelLoss(nn.Module):
    def __init__(self):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, psm, rm, pos_equal_one, neg_equal_one, targets):
        # psm (possibility score map): [batch_size, ac_rot_z = 2, H_map = 200, W_map = 176]
        # rm (regression map): [batch_size, ac_rot_z * encode_size = 14, H_map = 200, W_map = 176] 
        # pos_equal_one: [batch_size, H_map = 200, W_map = 176, ac_rot_z = 2]
        # neg_equal_one: [batch_size, H_map = 200, W_map = 176, ac_rot_z = 2]
        # targets: [batch_size, H_map = 200, W_map = 176, ac_rot_z * encode_size = 14]

        # p_pos: [batch_size, H_map, W_map, ac_rot_z]
        p_pos = torch.sigmoid(psm.permute(0, 2, 3, 1))

        # rm: [batch_size, H_map, W_map, ac_rot_z * encode_size]
        rm = rm.permute(0, 2, 3, 1).contiguous()

        # rm: [batch_size, H_map, W_map, ac_rot_z, encode_size]
        rm = rm.view(rm.size(0), rm.size(1), rm.size(2), -1, 7)

        # targets: [batch_size, H_map, W_map, ac_rot_z, encode_size]
        targets = targets.view(targets.size(0), targets.size(1), targets.size(2), -1, 7)

        # pos_equal_one_for_reg: [batch_size, H_map, W_map, ac_rot_z, encode_size]
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7)

        # rm_pos: [batch_size, H_map, W_map, ac_rot_z, encode_size]
        rm_pos = rm * pos_equal_one_for_reg

        # targets_pos: [batch_size, H_map, W_map, ac_rot_z, encode_size]
        targets_pos = targets * pos_equal_one_for_reg   

        cls_pos_loss = -pos_equal_one * torch.log(p_pos + cfg.eps)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + cfg.eps)

        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + cfg.eps)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + cfg.eps)

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + cfg.eps)
        conf_loss = cfg.alpha * cls_pos_loss + cfg.beta * cls_neg_loss        

        return conf_loss, reg_loss
