import numpy as np


class config:
    input_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/lyft-3d-od'

    model_name = 'batse_model'

    # points cloud ranges w.r.t car frame.
    # sample: b71497fc753ec107ca1ca6427f2513c550835aa244504550a5b0e2edd341f57d
    # x: -82.12028503417969 102.64202499389648
    # y: -78.11154389381409 84.82943511009216
    # z: -5.203974485397339 8.140005350112915
    xrange = (-100, 100)
    yrange = (-100, 100)
    zrange = (-5, 10)

    # voxel_size
    voxel_size = (0.2, 0.2, 0.5)

    num_epochs = 1
    batch_size = 16
    num_workers = 1
