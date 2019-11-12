import os
import sys
import time
import math
import pandas as pd
import numpy as np
import pylab as plt
import torch
import torch.nn
from scipy.spatial.transform import Rotation as R

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

from utils import utils
from utils.iou.box_overlaps import bbox_overlaps
from config import config as cfg
from data_loader import get_dataloader
from models import VoxelNet


class_name = 'car'
version = 'v4'
score_threshold = cfg.score_threshold

collision_iou_threshold = cfg.collision_iou_threshold
pretrained_model = 'model-{}-{}.ckpt'.format(class_name, version)
device = torch.device('cuda:0')

data_loaders, data_sizes = get_dataloader(phases=['test'])
model = VoxelNet().to(device)
checkpoint = torch.load(os.path.join(cfg.work_dir, 'data/models/pretrain', pretrained_model))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
sub = {}

since = time.time()
for idx, (voxel_features, voxel_coords, sample_tokens, ego_poses, calibrated_sensors) \
    in enumerate(data_loaders['test']):

    if idx % 1000 == 0:
        print('idx', idx)

    voxel_features = voxel_features.to(device)
    voxel_coords = voxel_coords.to(device)

    # Train and validate with one batch size.
    sample_token = sample_tokens[0]
    ego_pose = ego_poses[0]
    calibrated_sensor = calibrated_sensors[0]

    # psm: [batch_size, R_z, H_map, W_map]
    # rm: [batch_size, R_z * B_encode, H_map, W_map]
    psm, rm = model(voxel_features, voxel_coords, device)
    batch_size = psm.size(0)

    # psm: [batch_size, H_map, W_map, R_z]
    psm = torch.sigmoid(psm.permute(0,2,3,1))

    # psm: [batch_size, H_map * W_map * R_z]
    psm = psm.reshape((cfg.batch_size, -1))

    # rm: [batch_size, H_map, W_map, R_z * B_encode]
    rm = rm.permute(0,2,3,1).contiguous()

    # rm: [batch_size, H_map, W_map, R_z, R_z * B_encode]
    rm = rm.view(rm.size(0), rm.size(1), rm.size(2), 14)

    # prob: [batch_size, H_map * W_map * R_z]
    prob = psm.view(batch_size, -1)

    # batch_boxes3d: [batch_size, H_map * W_map * R_z, B_encode]
    batch_boxes3d = utils.delta_to_boxes3d(rm, device)

    mask = torch.gt(prob, score_threshold)

    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

    batch_id = 0

    # boxes3d: [H_map * W_map * R_z, B_encode]
    boxes3d = torch.masked_select(batch_boxes3d[batch_id], mask_reg[batch_id]).view(-1, 7)
    scores = torch.masked_select(prob[batch_id], mask[batch_id])

    # boxes2d_corners: [H_map * W_map * R_z, 4 corners, 2 xy]
    boxes3d = boxes3d.cpu().detach().numpy()
    boxes2d_corners = utils.boxes3d_to_corners(boxes3d)

    boxes2d = utils.boxes2d_four_corners_to_two_corners(boxes2d_corners)

    try:
        iou = bbox_overlaps(np.ascontiguousarray(boxes2d).astype(np.float32),
                            np.ascontiguousarray(boxes2d).astype(np.float32))

        scores = scores.cpu().detach().numpy()
        filter_idc = np.argmax((iou > collision_iou_threshold) * scores, axis=1)
        filter_idc = np.unique(filter_idc)
        scores = scores[filter_idc]                    # scores: [#pred_boxes, ]
        boxes3d = boxes3d[filter_idc]                  # boxes3d: [#pred_boxes, B_encode=7]
        boxes2d_corners = boxes2d_corners[filter_idc]  # boxes2d_corners: [#pred_boxes, 2 corners]

        boxes3d = utils.convert_boxes3d_xyzlwhr_to_Box(boxes3d)        
        boxes3d = utils.convert_boxes3d_from_sensor_to_global_frame(boxes3d, ego_pose, calibrated_sensor)

        for i in range(len(scores)):
            if math.isnan(boxes3d[i].orientation.yaw_pitch_roll[0]):
                continue
            if boxes3d[i].wlh[0] < 0.1 or boxes3d[i].wlh[0] > 10:
                continue
            if boxes3d[i].wlh[1] < 0.1 or boxes3d[i].wlh[1] > 10:
                continue
            if boxes3d[i].wlh[2] < 0.1 or boxes3d[i].wlh[2] > 10:
                continue

            pred = str(scores[i]) + ' ' + \
                   str(boxes3d[i].center[0]) + ' ' + \
                   str(boxes3d[i].center[1]) + ' ' + \
                   str(boxes3d[i].center[2]) + ' ' + \
                   str(boxes3d[i].wlh[0]) + ' ' + \
                   str(boxes3d[i].wlh[1]) + ' ' + \
                   str(boxes3d[i].wlh[2]) + ' ' + \
                   str(boxes3d[i].orientation.yaw_pitch_roll[0]) + ' ' + \
                   str(class_name) + ' '

            if sample_token in sub.keys():
                sub[sample_token] += pred
            else:
                sub[sample_token] = pred
    except ValueError:
        pred = '0.0001 2325.3639141771573 896.5050445940944 -17.888648146863467 1.8942416 4.564969 1.6333789 2.5251263265025563 car '
        if sample_token in sub.keys():
            sub[sample_token] += pred
        else:
            sub[sample_token] = pred

time_elapsed = time.time() - since
print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
      .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

sample_sub = pd.read_csv(os.path.join(cfg.work_dir, 'data/sample_submission.csv'))
sub = pd.DataFrame(list(sub.items()))
sub.columns = sample_sub.columns
sub.to_csv(os.path.join(cfg.work_dir, 'data/submissions/submission-' + class_name + '-' + version + '.csv'), index=False)
