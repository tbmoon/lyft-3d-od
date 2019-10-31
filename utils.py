import numpy as np
from lyft_dataset_sdk.utils.data_classes import Quaternion
from config import config as cfg


def move_boxes_to_car_frame(boxes3d, ego_pose):
    '''
    Move boxes from world frame to car frame.
    '''
    translation = -np.array(ego_pose['translation'])
    rotation = Quaternion(ego_pose['rotation']).inverse

    for box3d in boxes3d:
        box3d.translate(translation)
        box3d.rotate(rotation)
    return boxes3d


def filter_pointclouds_boxes3d(pointclouds, boxes3d=None):
    '''
    Filter pointclouds and/or boxes3d within a specific range.
    '''
    pxs = pointclouds[:, 0]
    pys = pointclouds[:, 1]
    pzs = pointclouds[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if boxes3d is not None:
        # True if at least one corner is within a specific range.
        box_corners_3d = []
        for box3d in boxes3d:
            # box_corner_3d: [3 xyz, 8 corners] -> [8 corners, 3 xyz]
            box_corner_3d = box3d.corners().transpose(1, 0)
            box_corners_3d.append(box_corner_3d)
        # box_corners_3d: [num_boxes, 8 corners, 3 xyz]
        box_corners_3d = np.array(box_corners_3d)
        box_x = (box_corners_3d[:, :, 0] >= cfg.xrange[0]) & (box_corners_3d[:, :, 0] < cfg.xrange[1])
        box_y = (box_corners_3d[:, :, 1] >= cfg.yrange[0]) & (box_corners_3d[:, :, 1] < cfg.yrange[1])
        box_z = (box_corners_3d[:, :, 2] >= cfg.zrange[0]) & (box_corners_3d[:, :, 2] < cfg.zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z, axis=1)
        return pointclouds[filter_xyz], box_corners_3d[box_xyz>0]

    return pointclouds[filter_xyz]
