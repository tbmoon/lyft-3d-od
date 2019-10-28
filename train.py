import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_dataloader
#from models import BaseModel

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    
    data_loaders = get_dataloader(
        input_dir=args.input_dir,
        phases=['train', 'valid'],
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od',
                        help='input directory for 3d object dectection competition.')

    parser.add_argument('--model_name', type=str, default='base_model',
                        help='base_model: VoxelNet.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size for training. (64)')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='the number of processes working on cpu. (16)')

    args = parser.parse_args()

    main(args)
