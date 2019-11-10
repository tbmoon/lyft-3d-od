import math
import numpy as np


class config:

    # Global parameters.
    input_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od'
    work_dir = '/home/mtb/ongoing_analysis/lyft-3d-od'
    model_name = 'base_model'
    load_model = True
    eps = 1e-6

    # The below ranges need to be determined after checking point cloud and box location w.r.t the sensor frame.
    xrange = (-80, 80)
    yrange = (-80, 80)
    zrange = (-4, 3)

    # Voxel size in meter.
    vox_width = 0.4
    vox_height = 0.4
    vox_depth = 0.8

    # Geometry of the anchor.
    # ac_length, ac_width, ac_height and ac_center_z in meter.
    ac_length = 3.9
    ac_width = 1.6
    ac_height = 1.56
    ac_center_z = -0.75
    ac_rot_z = 2

    # Maximum number of the point clouds in each voxel.
    pointclouds_per_vox = 45

    # IOU thresholds of positive and negative anchors. 
    iou_pos_threshold = 0.6
    iou_neg_threshold = 0.45

    # non-maximum suppression
    nms_threshold = 0.1
    score_threshold = 0.96

    # Loss parameters.
    alpha = 0.5
    beta = 1
    clip_grad_thres = 8.4

    # Optimizer parameters.
    learning_rate = 0.001
    step_size = 5
    gamma = 0.1
    accumulation_steps = 64

    # Training parameters.
    num_epochs = 20
    batch_size = 1
    num_workers = 16

    # W: number of voxels along x axis (no unit).
    # H: number of voxels along y axis (no unit).
    # D: number of voxels along z axis (no unit).
    W = math.ceil((xrange[1] - xrange[0]) / vox_width) 
    H = math.ceil((yrange[1] - yrange[0]) / vox_height)
    D = math.ceil((zrange[1] - zrange[0]) / vox_depth)

    # Not (W/2, H/2), but (H/2, W/2).
    # H/2 = H_map, W/2 = W_map.
    feature_map_shape = (int(H / 2), int(W / 2))

    # Pre-define 3D-anchor boxes with rotation around yaw axis: [H_map, W_map, ac_rot_z, encode_size].
    #   where, encode_size corrensponds to [x, y, z, l, w, h, rz].
    # Mapping voxel to voxel feature map.
    x = np.linspace(xrange[0]+vox_width, xrange[1]-vox_width, int(W/2))
    y = np.linspace(yrange[0]+vox_height, yrange[1]-vox_height, int(H/2))

    anchor_center_x, anchor_center_y = np.meshgrid(x, y)
    anchor_center_x = np.tile(anchor_center_x[..., np.newaxis], ac_rot_z)
    anchor_center_y = np.tile(anchor_center_y[..., np.newaxis], ac_rot_z)
    anchor_center_z = np.ones_like(anchor_center_x) * ac_center_z    

    anchor_length = np.ones_like(anchor_center_x) * ac_length
    anchor_width = np.ones_like(anchor_center_x) * ac_width
    anchor_height = np.ones_like(anchor_center_x) * ac_height

    anchor_rz = np.ones_like(anchor_center_x)
    anchor_rz[..., 0] = 0.
    anchor_rz[..., 1] = np.pi / 2

    anchors = np.stack([anchor_center_x,
                        anchor_center_y,
                        anchor_center_z,
                        anchor_length,
                        anchor_width,
                        anchor_height,
                        anchor_rz], axis=-1)
    anchors = anchors.reshape(-1, 7)
