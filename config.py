import os
import math
import numpy as np
import pandas as pd


class config:

    # Class counts in the lyft dataset:
    #   - animal                  186
    #   - bicycle               20928
    #   - bus                    8729
    #   - car                  534911
    #   - emergency_vehicle       132
    #   - motorcycle              818
    #   - other_vehicle         33376
    #   - pedestrian            24935
    #   - truck                 14164
    #
    # Select one of them!
    class_name = 'car'
    version = 'v2'

    # Global parameters.
    eps = 1e-6
    input_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od'
    work_dir = '/home/mtb/ongoing_analysis/lyft-3d-od'
    load_model = True
    model_name = 'model-{}-{}'.format(class_name, version)

    # The below ranges need to be determined after checking point cloud and box location w.r.t the sensor frame.
    xranges = {'car': (-75, 75),
               'pedestrian': (-55, 55)}
    yranges = {'car': (-75, 75),
               'pedestrian': (-55, 55)}
    xrange = xranges[class_name]
    yrange = yranges[class_name]
    zrange = (-4, 3)

    # Voxel size in meter.
    vox_widths  = {'car': 0.375,
                   'pedestrian': 0.275}
    vox_heights = {'car': 0.375,
                   'pedestrian': 0.275}
    vox_width = vox_widths[class_name]
    vox_height = vox_heights[class_name]
    vox_depth = 0.8

    # Geometry of the anchor.
    # ac_center_z in meter.
    ac_center_z = -0.75
    ac_rot_z = 2

    # Maximum number of the point clouds in each voxel.
    pointclouds_per_vox = 35

    # IOU thresholds of positive and negative anchors.
    # paper: iou threhold of car.
    #   - iou_pos_threshold = 0.6
    #   - iou_neg_threshold = 0.45
    iou_pos_thresholds = {'car': 0.85,
                          'pedestrian': 0.55}
    iou_neg_thresholds = {'car': 0.55,
                          'pedestrian': 0.25}
    iou_pos_threshold = iou_pos_thresholds[class_name]
    iou_neg_threshold = iou_neg_thresholds[class_name]

    # score_threshold for classification.
    score_thresholds = {'car': 0.999,
                        'pedestrian': 0.999}
    score_threshold = score_thresholds[class_name]

    # remove overlapping prediction.
    collision_iou_threshold = 0.2

    # Loss parameters.
    alpha = 0.5
    beta = 1
    clip_grad_thres = 8.4

    # Optimizer parameters.
    learning_rate = 0.0001
    step_size = 2
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

    # ac_length, ac_width and ac_height in meter are defined in data/mean_length_width_height.csv.
    df_mean_length_width_height = pd.read_csv(os.path.join(work_dir, 'data/mean_length_width_height.csv'))
    class_idc = df_mean_length_width_height['class_name'] == class_name
    ac_length = df_mean_length_width_height['length'][class_idc].values[0]
    ac_width = df_mean_length_width_height['width'][class_idc].values[0]
    ac_height = df_mean_length_width_height['height'][class_idc].values[0]

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
