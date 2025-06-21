import torch
import torch.nn.functional as F
import json
import os
import numpy as np
import imageio.v2 as imageio
# import matplotlib.pyplot as plt


def read_camera(folder):
    scene_info = json.load(open(os.path.join(folder, "info.json")))
    max_depth = 1.0
    try:
        max_depth = scene_info["images"][0]["max_depth"]
    except:
        pass

    rgb_files = []
    intrinsics = []  # focal_lenght and centre pixel
    poses = []

    scene_info["images"][0]["rgb"]

    for item in scene_info["images"]:
        rgb_files.append(os.path.join(folder, item["rgb"]))

        # coordinate system conversion
        c2w = item[
            "pose"
        ]  # pose tells us about the camera position and direction its looking at  pose gives rotation and translation

        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)

        poses.append(np.array(c2w_opencv))
        intrinsics.append(np.array(item["intrinsic"]))
    return rgb_files, poses, intrinsics, max_depth


def read_image(
    rgb_file, pose, src_intrinsic, max_depth, resize_factor=1, white_background=False
):
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
    intrinsic[:3, :3] = src_intrinsic

    if resize_factor != 1:
        image_size = image_size[0] * resize_factor, image_size[1] * resize_factor
        # if i resize the image i need to resize the intrinsic as well
        intrinsic[:2, :3] *= resize_factor
        resize_fn = lambda img, resize_factor: F.interpolate(
            img.permute(0, 3, 1, 2), scale_factor=resize_factor, mode="bilinear"
        ).permute(0, 2, 3, 1)  # 1st making BCWH format then revert back to BHWC

        rgb = resize_fn(rgb.unsqueeze(0), resize_factor).squeeze(0)
        depth = resize_fn(depth.unsqueeze(0), resize_factor).squeeze(0)
        alpha = resize_fn(alpha.unsqueeze(0), resize_factor).squeeze(0)

    camera = torch.from_numpy(
        np.concatenate((list(image_size), intrinsic.flatten(), pose.flatten())).astype(
            np.float32
        )
    )
    if white_background:
        rgb = alpha.unsqueeze(-1) * rgb + (1 - alpha.unsqueeze(-1))
    return rgb, depth, alpha, camera


# we are starting with implementation of read_all
def read_all(folder, resize_factor=1):
    src_rgb_files, src_poses, src_intrinsics, max_depth = read_camera(folder)

    src_cameras = []
    src_rgbs = []
    src_alphas = []
    src_depths = []

    resize_factor = 1
    for src_rgb_file, src_pose, src_intrinsic in zip(
        src_rgb_files, src_poses, src_intrinsics
    ):
        src_rgb, src_depth, src_alpha, src_camera = read_image(
            src_rgb_file,
            src_pose,
            src_intrinsic,
            resize_factor=resize_factor,
            max_depth=max_depth,
        )
        src_rgbs.append(src_rgb)
        src_depths.append(src_depth)
        src_alphas.append(src_alpha)
        src_cameras.append(src_camera)

    # stack converts list of torches to torch with dimension
    src_alphas = torch.stack(src_alphas, axis=0)
    src_depths = torch.stack(src_depths, axis=0)
    src_rgbs = torch.stack(src_rgbs, axis=0)
    src_cameras = torch.stack(src_cameras, axis=0)

    return {
        "rgb": src_rgbs[..., :3],
        "camera": src_cameras,
        "depth": src_depths,
        "alpha": src_alphas,
    }
