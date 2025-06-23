import torch
import json
import os
import numpy as np
import imageio.v2 as imageio


def scan_folder_info(folder):
    scene_info = json.load(open(os.path.join(folder, "info.json")))

    rgb_file_names = []
    all_data_intrinsics = []
    all_data_poses = []
    max_depth = []

    for item in scene_info["images"]:
        rgb_file_names.append(
            os.path.join(folder, item["rgb"])
        )  # so we got all the file names

        all_data_intrinsics.append(
            np.array(item["intrinsic"])
        )  # so we got all intrinsic data

        # finally we need all pose data

        c2w_blender = item["pose"]
        w2c_blender = np.linalg.inv(c2w_blender)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        all_data_poses.append(np.array(c2w_opencv))
        max_depth.append(item["max_depth"])

    # and finally we return all the data scanned from folder
    return rgb_file_names, all_data_intrinsics, all_data_poses, max_depth


def prepare_data(
    rgb_file_names, all_data_intrinsics, all_data_poses, all_data_max_depth
):
    all_data_cameras = []
    all_data_rgbs = []
    all_data_alphas = []
    all_data_depths = []

    for rgb_file, data_intrinsic, data_pose, max_depth in zip(
        rgb_file_names, all_data_intrinsics, all_data_poses, all_data_max_depth
    ):
        # read data from image files
        rgb = torch.from_numpy(imageio.imread(rgb_file).astype(np.float32) / 255.0)
        depth = torch.from_numpy(
            imageio.imread(rgb_file[:-7] + "depth.png").astype(np.float32)
            / 255.0
            * max_depth
        )
        alpha = torch.from_numpy(
            imageio.imread(rgb_file[:-7] + "alpha.png").astype(np.float32) / 255.0
        )

        image_size = rgb.shape[:2]
        intrinsic = np.eye(4, 4)

        intrinsic[:3, :3] = data_intrinsic

        camera = torch.from_numpy(
            np.concatenate(
                (list(image_size), intrinsic.flatten(), data_pose.flatten())
            ).astype(np.float32)
        )
        all_data_rgbs.append(rgb)
        all_data_depths.append(depth)
        all_data_alphas.append(alpha)
        all_data_cameras.append(camera)

    stacked_rgbs = torch.stack(all_data_rgbs, axis=0)
    stacked_alphas = torch.stack(all_data_alphas, axis=0)
    stacked_depths = torch.stack(all_data_depths, axis=0)
    stacked_cameras = torch.stack(all_data_cameras, axis=0)

    return {
        "rgb": stacked_rgbs[..., :3],
        "depth": stacked_depths,
        "alpha": stacked_alphas,
        "camera": stacked_cameras,
    }
