from gsplat.data_utils_eased import scan_folder_info, prepare_data
import torch
import numpy as np
from gsplat.point_utils import (
    PointCloud,
    parse_camera,
    make_rays_single_image,
    get_point_cloud,
)
from gsplat.guassModel import GuassModel


# --------------------#
device = "cuda"
folder = "./ref_repo/torch-splatting/B075X65R3X/"

rgb_file_names, intrinsics, poses, max_depth = scan_folder_info(folder)

data = prepare_data(rgb_file_names, intrinsics, poses, max_depth)

data = {k: v.to(device) for k, v in data.items()}

data["depth_range"] = torch.Tensor([[1, 3]] * len(data["rgb"])).to(device)

point_cloud = get_point_cloud(data)

sampled_point_cloud = point_cloud.random_sample(num_points=2**14)
# create_from_pcd(sampled_point_cloud)


gaussModel = GuassModel(max_sh=4)
gaussModel.create_from_pcd(pcd=sampled_point_cloud)
