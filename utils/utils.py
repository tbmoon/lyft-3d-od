import numpy as np
import torch
from lyft_dataset_sdk.utils.data_classes import Quaternion
from config import config as cfg


def delta_to_boxes3d(deltas, device):
    # Input:
    #   - deltas: [batch_size, H_map, W_map, R_z * B_encode = 14]
    # Ouput:
    #   - boxes3d: [batch_size, H_map * W_map * R_z, B_encode = 7]

    batch_size = deltas.shape[0]
    deltas = deltas.reshape(deltas.shape[0], -1, 7)  # deltas: [batch_size, H_map * W_map * R_z, B_encode]    
    anchors = torch.from_numpy(cfg.anchors).float().to(device)
    boxes3d = torch.zeros_like(deltas)

    anchors_d = torch.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)     # anchors_d: [H_map * W_map * R_z]
    anchors_d = anchors_d.repeat(batch_size, 2, 1).transpose(1, 2)  # anchors_d: [batch_size, H_map * W_map * R_z, B_encode]
    anchors = anchors.repeat(batch_size, 1, 1)                      # anchors: [batch_size, H_map * W_map * R_z, B_encode]

    # boxes3d: [batch_size, H_map * W_map * R_z, B_encode]
    boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors[..., [0, 1]]
    boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors[..., [5]]) + anchors[..., [2]]
    boxes3d[..., [3, 4, 5]] = torch.exp(deltas[..., [3, 4, 5]]) * anchors[..., [3, 4, 5]]
    boxes3d[..., 6] = torch.asin(deltas[..., 6]) + anchors[..., 6]

    return boxes3d


def boxes3d_to_corners(boxes3d, rotate=True):
    # input:
    #   - boxes3d: [H_map * W_map * R_z, B_encode]
    # output:
    #   - boxes2d_corner: [H_map * W_map * R_z, 4 corners, 2 xy]

    num_boxes = boxes3d.shape[0]
    l, w, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]

    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T  # (N, 4)
    y_corners = np.array([-w / 2., w / 2., w / 2., -w / 2.], dtype=np.float32).T  # (N, 4)

    if rotate:
        rz = boxes3d[:, 6]
        rot_list = np.array([[ np.cos(rz), np.sin(rz)],
                             [-np.sin(rz), np.cos(rz)]])  # (2, 2, N)
        R_list = np.transpose(rot_list, (2, 0, 1))        # (N, 2, 2)

        temp_corners = np.concatenate((x_corners.reshape(-1, 4, 1), y_corners.reshape(-1, 4, 1)), axis=2)  # (N, 4, 2)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 4, 2)
        x_corners, y_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1]

    x_loc, y_loc = boxes3d[:, 0], boxes3d[:, 1]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 4)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 4)

    corners = np.concatenate((x.reshape(-1, 4, 1), y.reshape(-1, 4, 1)), axis=2)

    return corners.astype(np.float32)


def gt_boxes3d_center_to_bottom_corner(gt_boxes3d):
    num_boxes = len(gt_boxes3d)
    gt_boxes_bottom_corner = np.zeros((num_boxes, 4, 2))  # [num_boxes, 4 bottom corners, 2 xy]    
    for i, gt_box3d in enumerate(gt_boxes3d):
        gt_box_bottom_corner = gt_box3d.bottom_corners()  # [3 xyz, 4 bottom corners]
        gt_box_bottom_corner = gt_box_bottom_corner.transpose(1, 0)[:, :2]  # [4 bottom corners, 2 xy]
        gt_boxes_bottom_corner[i] = gt_box_bottom_corner

    return gt_boxes_bottom_corner


def boxes2d_four_corners_to_two_corners(boxes2d_corner):
    # boxes2d_corner: [num_boxes, 4 corners, 2 xy]
    # boxes2d_two_corner: [num_boxes, 4]
    #   where, 4 means (2 xy) * (2 corners): x_min, y_min, x_max, y_max
    assert boxes2d_corner.ndim == 3
    num_boxes = boxes2d_corner.shape[0]
    boxes2d_two_corner = np.zeros((num_boxes, 4))
    boxes2d_two_corner[:, 0] = np.min(boxes2d_corner[:, :, 0], axis=1)
    boxes2d_two_corner[:, 1] = np.min(boxes2d_corner[:, :, 1], axis=1)
    boxes2d_two_corner[:, 2] = np.max(boxes2d_corner[:, :, 0], axis=1)
    boxes2d_two_corner[:, 3] = np.max(boxes2d_corner[:, :, 1], axis=1)

    return boxes2d_two_corner


def convert_gt_boxes3d_from_global_to_sensor_frame(gt_boxes3d, ego_pose, calibrated_sensor):
    '''
    Convert gt_boxes3d from the global frame to the sensor frame.
    '''
    # From the global frame to the car frame.
    for gt_box3d in gt_boxes3d:
        gt_box3d.translate(-np.array(ego_pose['translation']))
        gt_box3d.rotate(Quaternion(ego_pose['rotation']).inverse)

    # From the car frame to the sensor frame.
    for gt_box3d in gt_boxes3d:
        gt_box3d.translate(-np.array(calibrated_sensor["translation"]))
        gt_box3d.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

    return gt_boxes3d


def filter_pointclouds_gt_boxes3d(pointclouds, gt_boxes3d=None, class_name=None):
    '''
    Filter pointclouds and/or gt_boxes3d within a specific range.
    '''
    pxs = pointclouds[:, 0]
    pys = pointclouds[:, 1]
    pzs = pointclouds[:, 2]
    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)
    pointclouds = pointclouds[filter_xyz]

    if gt_boxes3d is not None:
        # True if at least one corner is within a specific range.
        gt_boxes3d_corner = []
        for gt_box3d in gt_boxes3d:
            # gt_box3d_corner: [3 xyz, 8 corners] -> [8 corners, 3 xyz]
            gt_box3d_corner = gt_box3d.corners().transpose(1, 0)
            gt_boxes3d_corner.append(gt_box3d_corner)
        # gt_boxes3d_corners: [num_boxes, 8 corners, 3 xyz]
        gt_boxes3d_corner = np.array(gt_boxes3d_corner)
        box_x = (gt_boxes3d_corner[:, :, 0] >= cfg.xrange[0]) & (gt_boxes3d_corner[:, :, 0] < cfg.xrange[1])
        box_y = (gt_boxes3d_corner[:, :, 1] >= cfg.yrange[0]) & (gt_boxes3d_corner[:, :, 1] < cfg.yrange[1])
        box_z = (gt_boxes3d_corner[:, :, 2] >= cfg.zrange[0]) & (gt_boxes3d_corner[:, :, 2] < cfg.zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z, axis=1)
        box_xyz = np.array([i for i, val in enumerate(box_xyz) if val], dtype=int)
        gt_boxes3d = [gt_boxes3d[i] for i in box_xyz if gt_boxes3d[i].name == class_name]

        return pointclouds, gt_boxes3d

    return pointclouds
