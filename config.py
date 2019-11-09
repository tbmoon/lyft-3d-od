import math
import numpy as np


class config:
    input_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od'
    work_dir = '/home/mtb/ongoing_analysis/lyft-3d-od'
    model_name = 'base_model'

    eps = 1e-6

    # The below ranges are determined after checking point cloud w.r.t the sensor frame.
    # Update this after checking point clouds along with box position using all samples.
    #xrange = (-100, 100)
    #yrange = (-100, 100)
    #zrange = (-10, 10)
    xrange = (-70.4, 70.4)
    yrange = (-80, 80)
    zrange = (-3, 1)

    # Voxel size.
    vox_width = 0.4
    vox_height = 0.4
    vox_depth = 0.4

    # Geometry of the anchor.
    ac_length = 3.9
    ac_width = 1.6
    ac_height = 1.56
    ac_center_z = 0.
    ac_rot_z = 2

    # Maximum number of the point clouds in each voxel.
    pointclouds_per_vox = 35

    # IOU thresholds of positive and negative anchors. 
    iou_pos_threshold = 0.6
    iou_neg_threshold = 0.45

    # non-maximum suppression
    nms_threshold = 0.1
    score_threshold = 0.96
    
    # Loss parameters.
    alpha = 1
    beta = 10
    reg = 3
    focal_loss_gamma = 5
    clip_grad_thres = 8.4

    # Optimizer parameters.
    learning_rate = 0.001
    step_size = 5
    gamma = 0.1
    accumulation_steps = 64

    num_epochs = 20
    batch_size = 1
    num_workers = 16


    # W: number of voxels along x axis.
    # H: number of voxels along y axis.
    # D: number of voxels along z axis.
    W = math.ceil((xrange[1] - xrange[0]) / vox_width) 
    H = math.ceil((yrange[1] - yrange[0]) / vox_height)
    D = math.ceil((zrange[1] - zrange[0]) / vox_depth)

    # Not (W/2, H/2), but (H/2, W/2).
    feature_map_shape = (int(H / 2), int(W / 2))

    # anchors: [200, 176, 2, 7] x y z l w h r
    # Mapping voxel to voxel feature map.
    x = np.linspace(xrange[0]+vox_width, xrange[1]-vox_width, int(W/2))
    y = np.linspace(yrange[0]+vox_height, yrange[1]-vox_height, int(H/2))

    # Pre-define anchor boxes.
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
