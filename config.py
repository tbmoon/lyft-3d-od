import numpy as np


class config:
    input_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od'

    model_name = 'batse_model'

    # Some hyperparameters we'll need to define for the system.
    voxel_size = (0.4, 0.4, 1.5)
    z_offset = -2.0
    bev_shape = (336, 336, 3)

    batch_size = 16
    num_workers = 16
