import math
import numpy as np


class config:
    input_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od'

    model_name = 'base_model'

    # The below ranges are determined after checking point cloud w.r.t the sensor frame.
    # Update this after checking point clouds along with box position using all samples.
    #xrange = (-100, 100)
    #yrange = (-100, 100)
    #zrange = (-10, 10)
    xrange = (-70.4, 70.4)
    yrange = (-80, 80)
    zrange = (-4, 4)

    # Voxel size.
    vox_width = 0.4
    vox_height = 0.4
    vox_depth = 0.8

    # W: number of voxels along x axis.
    # H: number of voxels along y axis.
    # D: number of voxels along z axis.
    W = math.ceil((xrange[1] - xrange[0]) / vox_width) 
    H = math.ceil((yrange[1] - yrange[0]) / vox_height)
    D = math.ceil((zrange[1] - zrange[0]) / vox_depth)

    # Geometry of the anchor.
    anchor_length = 3.9
    anchor_width = 1.6
    anchor_height = 1.56
    anchor_center_z = 0.
    anchor_two_rotations = 2

    # Maximum number of the point clouds in each voxel.
    pointclouds_per_vox = 70

    # IOU thresholds of positive and negative anchors. 
    iou_pos_threshold = 0.6
    iou_neg_threshold = 0.45

    # Loss parameters.
    alpha = 1
    beta = 10
    reg = 3
    clip_grad_thres = 8.4

    # Optimizer parameters.
    learning_rate = 0.001
    step_size = 5
    gamma = 0.1

    num_epochs = 20
    batch_size = 1
    num_workers = 16
