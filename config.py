import numpy as np


class config:
    input_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od'

    model_name = 'batse_model'

    # The below ranges are determined after checking point cloud w.r.t the sensor frame.
    # Update this after checking point clouds along with box position using all samples.
    xrange = (-100, 100)
    yrange = (-100, 100)
    zrange = (-10, 10)

    # Voxel size.
    vox_width = 0.2
    vox_height = 0.2
    vox_depth = 0.5

    # Geometry of the anchor.
    anchor_length = 3.9
    anchor_width = 1.6
    anchor_height = 1.56
    anchor_center_z = 0.
    anchor_two_rotations = 2

    # Maximum number of the point clouds in each voxel.
    pointclouds_per_vox = 40

    # Number of anchors for each voxel feature.
    anchors_per_vox_feat = 2

    # IOU thresholds of positive and negative anchors. 
    iou_pos_threshold = 0.6
    iou_neg_threshold = 0.45
    

    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    num_epochs = 1
    batch_size = 16
    num_workers = 1
