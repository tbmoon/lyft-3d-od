import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from config import config as cfg

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class Conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,k,s,p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=k,stride=s,padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation
    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.activation:
            return F.relu(x,inplace=True)
        else:
            return x

class FCN(nn.Module):

    def __init__(self,cin,cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self,x):
        # KK is the stacked k across batch
        K, T, _ = x.shape
        x = self.linear(x.view(K*T,-1))
        x = self.bn(x)
        x = F.relu(x)
        return x.view(K,T,-1)


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self,cin,cout):
        super(VFE, self).__init__()
        self.units = int(cout / 2)
        self.fcn = FCN(cin,self.units)

    def forward(self, x, mask):
        # point-wise feauture
        pointwise = self.fcn(x)
        aggregated = torch.max(pointwise,1)[0]
        repeated = aggregated.unsqueeze(1).repeat(1,cfg.pointclouds_per_vox,1)
        pointwise_concat = torch.cat((pointwise,repeated),dim=2)
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        return pointwise_concat * mask.float()


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):
    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7,32)
        self.vfe_2 = VFE(32,128)
        self.fcn = FCN(128,128)
    def forward(self, x):
        mask = torch.ne(torch.max(x,2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x,1)[0]
        return x

# Convolutional Middle Layer
class ConvoMidLayer(nn.Module):
    def __init__(self):
        super(ConvoMidLayer, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x

# Region Proposal Network
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = nn.Sequential(Conv2d(128, 128, 3, 2, 1),
                                     Conv2d(128, 128, 3, 1, 1),
                                     Conv2d(128, 128, 3, 1, 1),
                                     Conv2d(128, 128, 3, 1, 1))
       
        self.block_2 = nn.Sequential(Conv2d(128, 128, 3, 2, 1),
                                     Conv2d(128, 128, 3, 1, 1),
                                     Conv2d(128, 128, 3, 1, 1),
                                     Conv2d(128, 128, 3, 1, 1),
                                     Conv2d(128, 128, 3, 1, 1),
                                     Conv2d(128, 128, 3, 1, 1))

        self.block_3 = nn.Sequential(Conv2d(128, 256, 3, 2, 1),
                                     nn.Conv2d(256, 256, 3, 1, 1),
                                     nn.Conv2d(256, 256, 3, 1, 1),
                                     nn.Conv2d(256, 256, 3, 1, 1),
                                     nn.Conv2d(256, 256, 3, 1, 1),
                                     nn.Conv2d(256, 256, 3, 1, 1))

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),
                                      nn.BatchNorm2d(256))
        
        self.score_head = Conv2d(768, cfg.anchor_two_rotations, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * cfg.anchor_two_rotations, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self,x):
        x = self.block_1(x)
        x_skip_bloc1 = x
        x = self.block_2(x)
        x_skip_bloc2 = x
        x = self.block_3(x)

        deconv_1 = self.deconv_1(x)
        deconv_2 = self.deconv_2(x_skip_bloc2)
        deconv_3 = self.deconv_3(x_skip_bloc1)

        x = torch.cat((deconv_1, deconv_2, deconv_3), 1)
        return self.score_head(x),self.reg_head(x)

class VoxelNet(nn.Module):
    def __init__(self):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = ConvoMidLayer()
        self.rpn = RPN()

    def voxelize(self, sparse_features, coords):
        dim = sparse_features.shape[-1]
        dense_feature = torch.zeros(dim, cfg.batch_size, cfg.D, cfg.H, cfg.W).cuda()
        dense_feature[:, coords[:,0], coords[:,1], coords[:,2], coords[:,3]]= sparse_features.transpose(1,0)
        return dense_feature.transpose(0,1)

    def forward(self, voxel_features, voxel_coords):
        # feature learning network
        vwfs = self.svfe(voxel_features)
        vwfs = self.voxelize(vwfs, voxel_coords)
        psm,rm = self.rpn(self.cml(vwfs).view(cfg.batch_size, -1, cfg.H, cfg.W))
        return psm, rm
