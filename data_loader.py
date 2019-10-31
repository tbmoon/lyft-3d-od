import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix

from config import config as cfg
import utils


class LyftLevel5Dataset(data.Dataset):
    def __init__(self,
                 input_dir,
                 phase):
        self.df = pd.read_csv(os.path.join(os.getcwd(), 'data', phase + '.csv'))
        self.lyft_dataset = LyftDataset(data_path=os.path.join(input_dir, 'train'),
                                        json_path=os.path.join(input_dir, 'train', 'data'),
                                        verbose=False)
        self.xrange = cfg.xrange
        self.yrange = cfg.yrange
        self.zrange = cfg.zrange
        self.voxel_size = cfg.voxel_size
        self.max_pointclouds = cfg.max_pointclouds

    def voxelize_pointclouds(self, pointclouds):
        # Shuffle the pointclouds.
        np.random.shuffle(pointclouds)

        # x, y and z correspond to width, height and depth, respectively.
        voxel_coords = ((pointclouds[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
            self.voxel_size[0], self.voxel_size[1], self.voxel_size[2])).astype(np.int32)

        # Convert to (z, y, x).
        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords,
                                                        axis=0,
                                                        return_inverse=True,
                                                        return_counts=True)

        # Fill each voxel with voxel feature.
        voxel_features = []
        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.max_pointclouds, 7), dtype=np.float32)
            pts = pointclouds[inv_ind == i]
            if voxel_counts[i] > self.max_pointclouds:
                pts = pts[:self.max_pointclouds, :]
                voxel_counts[i] = self.max_pointclouds
            # Augment the points.
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        return np.array(voxel_features), voxel_coords

    def __getitem__(self, idx):
        sample = self.lyft_dataset.get('sample', self.df['sample_token'][idx])
        lidar = self.lyft_dataset.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_data_path = self.lyft_dataset.get_sample_data_path(sample['data']['LIDAR_TOP'])        
        boxes3d = self.lyft_dataset.get_boxes(sample['data']['LIDAR_TOP'])

        ego_pose = self.lyft_dataset.get('ego_pose', lidar['ego_pose_token'])
        calibrated_sensor = self.lyft_dataset.get('calibrated_sensor', lidar['calibrated_sensor_token'])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']),
                                           inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                           Quaternion(calibrated_sensor['rotation']),
                                           inverse=False)

        # pointclouds w.r.t sensor frame: [4 xyzi, num_points]
        pointclouds = LidarPointCloud.from_file(lidar_data_path)
        # pointclouds w.r.t car frame.
        pointclouds.transform(car_from_sensor)
        # pointclouds: [4 xyzi, num_points] -> [num_points, 4 xyzi]
        pointclouds = pointclouds.points.transpose(1, 0)
        
        # Convert boxes3d from world frame to car frame.
        boxes3d = utils.move_boxes_to_car_frame(boxes3d, ego_pose)
        
        # Data augmentation.
        #lidar, gt_box3d = aug_data(lidar, gt_box3d)

        # Filter point clouds and boxes within a specific range.
        # boxes3d: [num_boxes, 8 corners, 3 xyz]
        pointclouds, boxes3d = utils.filter_pointclouds_boxes3d(pointclouds, boxes3d)

        # Voxelize point clouds.
        voxel_features, voxel_coords = self.voxelize_pointclouds(pointclouds)
        return voxel_features, voxel_coords

    def __len__(self):
        return 1
        #return len(self.df)


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
