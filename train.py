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
#from models import BaseModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():

    data_loaders = get_dataloader(
        input_dir=cfg.input_dir,
        phases=['train', 'valid'],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers)
    
    
    for epoch in range(cfg.num_epochs):
        for phase in ['train', 'valid']:
            since = time.time()
            
            if phase == 'train':
                pass
            #model.train()
            else:
                pass
            #model.eval()
            for idx, (pointcloud) in enumerate(data_loaders[phase]):
                print(idx)
                #optimizer.zero_grad()

if __name__ == '__main__':
    main()
