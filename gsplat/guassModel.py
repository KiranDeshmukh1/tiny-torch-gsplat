import torch
import numpy as np
import torch.nn as nn
from gsplat.point_utils import PointCloud
from simple_knn._C import distCUDA2


class GuassModel(nn.Module):
    def __init__(self, max_sh):
        super().__init__()
        self.max_sh = max_sh
        self.debug = False
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_ac = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        print("GaussModel initialized")

    def create_from_pcd(self, pcd: PointCloud):
        C0 = 0.28209479177387814
        RGB2SH = lambda rgb: ((rgb - 0.5) / C0)
        points = pcd.coords
        color_channels = pcd.select_channels(["R", "G", "B"]) / 255.0
        coordinates = torch.tensor(np.asarray(points)).float()

        colors = RGB2SH(torch.tensor(np.asarray(color_channels))).float().cuda()

        print("Number of points at initialisation : ", coordinates.shape[0])

        features = (
            torch.zeros(colors.shape[0], 3, (self.max_sh + 1) ** 2).float().cuda()
        )
        features[:, :3, 0] = colors
        features[:, :3, 1:] = 0.0
        point_knn_mean_dist = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001
        )

        scales = torch.log(torch.sqrt(point_knn_mean_dist))[..., None].repeat(1, 3)
        rotations = torch.zeros(coordinates.shape[0], 4).cuda()
        rotations[:, 0] = 1.0

        # we are implementing inverse sigmoid so lets go into guassrendere
        inverse_sigmoid = lambda x: torch.log(x / (1 - x))

        opacities = inverse_sigmoid(
            0.1
            * torch.ones((coordinates.shape[0], 1), dtype=torch.float32, device="cuda")
        )

        if self.debug:
            color_channels = np.zeros_like(color_channels)
            opacities = inverse_sigmoid(
                0.9
                * torch.ones(
                    (coordinates.shape[0], 1), dtype=torch.float32, device="cuda"
                )
            )

        self._xyz = nn.Parameter(coordinates.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )

        self._features_ac = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        print("done")
        return self
