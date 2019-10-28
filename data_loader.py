import os
import numpy as np
import torch
import torch.utils.data as data

from lyft_dataset_sdk.lyftdataset import LyftDataset


class LyftLevel5Dataset(data.Dataset):
    def __init__(self,
                 input_dir,
                 phase):
        
        self.lyft_dataset = LyftDataset(data_path=os.path.join(input_dir, 'train'),
                                        json_path=os.path.join(input_dir, 'train', 'data'),
                                        verbose=True)


    def __getitem__(self, idx):
        pass

    def __len__(self):
        return 1


def get_dataloader(
    input_dir,
    phases,
    batch_size,
    num_workers):

    lyftlevel5_datasets = {
        phase: LyftLevel5Dataset(
            input_dir=input_dir,
            phase=phase)
        for phase in phases}

    data_loaders = {
        phase: torch.utils.data.DataLoader(
            dataset=lyftlevel5_datasets[phase],
            batch_size=batch_size,
            shuffle=True if phase is not 'test' else False,
            num_workers=num_workers,
            pin_memory=False)
        for phase in phases}

    return data_loaders
