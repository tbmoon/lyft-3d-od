import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

#from cython_bbox import bbox_overlaps
from box_overlaps import bbox_overlaps
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

        # Define voxel space.
        # *- Convention -*
        #   - x @ reference frame: vox_width
        #   - y @ reference frame: vox_height
        #   - z @ reference frame: vox_depth
        self.xrange = cfg.xrange
        self.yrange = cfg.yrange
        self.zrange = cfg.zrange
        self.vox_width = cfg.vox_width
        self.vox_height = cfg.vox_height
        self.vox_depth = cfg.vox_depth

        # Pre-defined anchors.
        # *- Convention -*
        #   - x @ box frame points forward    : anchor_length
        #   - y @ box frame points to the left: anchor_width
        #   - z @ box frame points up         : anchor_height
        self.anchor_length = cfg.anchor_length
        self.anchor_width = cfg.anchor_width
        self.anchor_height = cfg.anchor_height
        self.anchor_center_z = cfg.anchor_center_z
        self.anchor_two_rotations = cfg.anchor_two_rotations

        # Number of point clouds sampled in each voxel.
        self.pointclouds_per_vox = cfg.pointclouds_per_vox

        # IOU depends on class size.
        self.iou_pos_threshold = cfg.iou_pos_threshold
        self.iou_neg_threshold = cfg.iou_neg_threshold

        # W: number of voxels along x axis.
        # H: number of voxels along y axis.
        self.W = math.ceil((self.xrange[1] - self.xrange[0]) / self.vox_width) 
        self.H = math.ceil((self.yrange[1] - self.yrange[0]) / self.vox_height)

        # Not (W/2, H/2), but (H/2, W/2).
        self.feature_map_shape = (int(self.H / 2), int(self.W / 2))

        # Mapping voxel to voxel feature map.
        x = np.linspace(self.xrange[0]+self.vox_width, self.xrange[1]-self.vox_width, int(self.W/2))
        y = np.linspace(self.yrange[0]+self.vox_height, self.yrange[1]-self.vox_height, int(self.H/2))

        # Pre-define anchor boxes.
        anchor_center_x, anchor_center_y = np.meshgrid(x, y)
        anchor_center_x = np.tile(anchor_center_x[..., np.newaxis], self.anchor_two_rotations)
        anchor_center_y = np.tile(anchor_center_y[..., np.newaxis], self.anchor_two_rotations)

        shape = np.shape(anchor_center_x)
        anchor_length = np.ones(shape) * self.anchor_length
        anchor_width = np.ones(shape) * self.anchor_width
        anchor_height = np.ones(shape) * self.anchor_height
        anchor_center_z = np.ones(shape) * self.anchor_center_z
        anchor_yaw = np.ones(shape)
        anchor_yaw[..., 0] = 0.
        anchor_yaw[..., 1] = np.pi / 2

        self.anchors = np.stack([anchor_center_x,
                                 anchor_center_y,
                                 anchor_center_z,
                                 anchor_length,
                                 anchor_width,
                                 anchor_height,
                                 anchor_yaw], axis=-1)
        self.anchors = self.anchors.reshape(-1, 7)

    def voxelize_pointclouds(self, pointclouds):
        # Shuffle the pointclouds.
        np.random.shuffle(pointclouds)

        # x, y and z correspond to vox_width, vox_height and vox_depth, respectively.
        voxel_coords = ((pointclouds[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
            self.vox_width, self.vox_height, self.vox_depth)).astype(np.int32)

        # Convert to (z, y, x) to sample unique voxel.
        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords,
                                                        axis=0,
                                                        return_inverse=True,
                                                        return_counts=True)

        # Fill each voxel with voxel feature.
        voxel_features = []
        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.pointclouds_per_vox, 7), dtype=np.float32)
            pts = pointclouds[inv_ind == i]
            if voxel_counts[i] > self.pointclouds_per_vox:
                pts = pts[:self.pointclouds_per_vox, :]
                voxel_counts[i] = self.pointclouds_per_vox
            # Augment the points: (x, y, z) -> (x, y, z, i, x-mean, y-mean, z-mean).
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        return np.array(voxel_features), voxel_coords

    def cal_target(self, gt_boxes3d):
        anchors_d = np.sqrt(self.anchors[:, 3]**2 + self.anchors[:, 4]**2)
        pos_equal_one = np.zeros((*self.feature_map_shape, self.anchor_two_rotations))
        neg_equal_one = np.zeros((*self.feature_map_shape, self.anchor_two_rotations))
        targets = np.zeros((*self.feature_map_shape, 7 * self.anchor_two_rotations))

        gt_boxes3d_xyzlwhr = [[gt_box3d.center[0],
                               gt_box3d.center[1],
                               gt_box3d.center[2],
                               gt_box3d.wlh[1],
                               gt_box3d.wlh[0],
                               gt_box3d.wlh[2],
                               gt_box3d.orientation.radians] for gt_box3d in gt_boxes3d]

        # Only taken into account rotation around yaw axis.
        # It means bottom-corners equal to top-corner.
        anchors_bottom_corner = utils.anchors_center_to_bottom_corner(self.anchors)
        gt_boxes_bottom_corner = utils.gt_boxes3d_center_to_bottom_corner(gt_boxes3d)

        anchor_boxes2d = utils.boxes2d_four_corners_to_two_corners(anchors_bottom_corner)
        gt_boxes2d = utils.boxes2d_four_corners_to_two_corners(gt_boxes_bottom_corner) 

        #iou = bbox_overlaps(np.ascontiguousarray(anchor_boxes2d, dtype=np.float32),
        #                    np.ascontiguousarray(gt_boxes2d, dtype=np.float32))
        
        iou = bbox_overlaps(np.ascontiguousarray(anchor_boxes2d).astype(np.float32),
                            np.ascontiguousarray(gt_boxes2d).astype(np.float32))
        
        print(iou)
        
        
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        
    
    def __getitem__(self, idx):
        # Point clouds of lidar are positioned at the sensor frame,
        # while gt_boxes3d at the global frame.
        sample = self.lyft_dataset.get('sample', self.df['sample_token'][idx])
        lidar_info = self.lyft_dataset.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_data_path = self.lyft_dataset.get_sample_data_path(sample['data']['LIDAR_TOP'])        
        gt_boxes3d = self.lyft_dataset.get_boxes(sample['data']['LIDAR_TOP'])

        ego_pose = self.lyft_dataset.get('ego_pose', lidar_info['ego_pose_token'])
        calibrated_sensor = self.lyft_dataset.get('calibrated_sensor', lidar_info['calibrated_sensor_token'])

        # pointclouds w.r.t. the sensor frame: [4 xyzi, num_points]
        pointclouds = LidarPointCloud.from_file(lidar_data_path)
        # pointclouds: [4 xyzi, num_points] -> [num_points, 4 xyzi]
        pointclouds = pointclouds.points.transpose(1, 0)

        # Convert gt_boxes3d from the global frame to the sensor frame.
        gt_boxes3d = utils.convert_gt_boxes3d_from_global_to_sensor_frame(gt_boxes3d, ego_pose, calibrated_sensor)

        # Data augmentation.
        #pointclouds, gt_box3d = aug_data(pointclouds, gt_box3d)

        # Filter point clouds and gt_boxes3d within a specific range.
        pointclouds, gt_boxes3d = utils.filter_pointclouds_gt_boxes3d(pointclouds, gt_boxes3d)

        # Voxelize point clouds.
        voxel_features, voxel_coords = self.voxelize_pointclouds(pointclouds)

        # Encode bounding boxes.
        pos_equal_one, neg_equal_one, targets = self.cal_target(gt_boxes3d)

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
