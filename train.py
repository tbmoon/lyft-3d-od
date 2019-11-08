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
from loss import VoxelLoss

from config import config as cfg
from models import VoxelNet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    os.makedirs(os.path.join(os.getcwd(), 'data/logs'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'data/models'), exist_ok=True)

    data_loaders, data_sizes = get_dataloader(
        phases=['train', 'valid'])

    model = VoxelNet().to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = VoxelLoss(alpha=cfg.alpha, beta=cfg.beta, reg=cfg.reg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    for epoch in range(cfg.num_epochs):
        for phase in ['train', 'valid']:
            since = time.time()
            running_conf_loss = 0.0
            running_reg_loss = 0.0
            running_total_loss = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            optimizer.zero_grad()
            for idx, (voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets) \
                in enumerate(data_loaders[phase]):

                voxel_features = voxel_features.to(device)
                pos_equal_one = pos_equal_one.to(device)
                neg_equal_one = neg_equal_one.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    psm, rm = model(voxel_features, voxel_coords)

                    conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
                    total_loss = conf_loss + reg_loss

                    if phase == 'train':
                        total_loss.backward()
                        #_ = nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_thres)

                        if (idx + 1) % cfg.accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    running_conf_loss += conf_loss.item()
                    running_reg_loss += reg_loss.item()
                    running_total_loss += total_loss.item()

            epoch_conf_loss = running_conf_loss / data_sizes[phase]
            epoch_reg_loss = running_reg_loss / data_sizes[phase]
            epoch_total_loss = running_total_loss / data_sizes[phase]

            print('| {} SET | Epoch [{:02d}/{:02d}]'.format(phase.upper(), epoch+1, cfg.num_epochs))
            print('\t*- Conf. Loss        : {:.4f}'.format(epoch_conf_loss))
            print('\t*- Reg. Loss         : {:.4f}'.format(epoch_reg_loss))
            print('\t*- Total Loss        : {:.4f}'.format(epoch_total_loss))
            
            # Log the loss in an epoch.
            with open(os.path.join(os.getcwd(), 'data/logs/{}-log-epoch-{:02}.txt').format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t' +
                        str(epoch_conf_loss) + '\t' +
                        str(epoch_reg_loss) + '\t' +
                        str(epoch_total_loss))

            # Save the model check points.
            if phase == 'train':
                torch.save({'epoch': epoch+1,
                            'state_dict': model.state_dict()},
                           os.path.join(os.getcwd(), 'data/models/model-epoch-{:02d}.ckpt'.format(epoch+1)))

            time_elapsed = time.time() - since
            print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
                  .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
        scheduler.step()
        print()


if __name__ == '__main__':
    main()
