import os
import time
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_dataloader

from config import config as cfg
from models import VoxelNet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():

    data_loaders = get_dataloader(
        input_dir=cfg.input_dir,
        phases=['train', 'valid'],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers)
    
    model = VoxelNet().to(device)
    
    #criterion = nn.CrossEntropyLoss()

    #params = list(model.img_encoder.fc.parameters()) \
    #    + list(model.qst_encoder.parameters()) \
    #    + list(model.fc1.parameters()) \
    #    + list(model.fc2.parameters())

    #optimizer = optim.Adam(params, lr=cfg.learning_rate)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    for epoch in range(cfg.num_epochs):
        for phase in ['train', 'valid']:
            since = time.time()
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for idx, (voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets) in enumerate(data_loaders[phase]):
                if idx % 100 == 0:
                    print("idx", idx)

                voxel_features = voxel_features.to(device)
                pos_equal_one = pos_equal_one.to(device)
                neg_equal_one = neg_equal_one.to(device)
                targets = targets.to(device)

                psm, rm = model(voxel_features, voxel_coords)
                
            time_elapsed = time.time() - since
            print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
                  .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))


if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
