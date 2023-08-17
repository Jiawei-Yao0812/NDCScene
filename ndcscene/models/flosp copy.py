import torch
import torch.nn as nn
import torch.nn.functional as F


class FLoSP(nn.Module):
    def __init__(self, scene_size, dataset, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x, projected_pix):
        b, c, d, h, w = x[0].shape
        if self.dataset == "NYU":
            projected_pix_3d = projected_pix.reshape(
                b,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                3,
            )
            projected_pix_2d = projected_pix.reshape(
                b,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale * self.scene_size[1] // self.project_scale,
                3,
            )[:, :, :, :2]
            x_3d = sum([F.grid_sample(_, projected_pix_3d, align_corners=True) for _ in x if _.ndim == 5])
            x_2d = sum([F.grid_sample(_, projected_pix_2d, align_corners=True).reshape(*x_3d.shape) for _ in x if _.ndim == 4])
            x = x_2d + x_3d
            x = x.permute(0, 1, 2, 4, 3)
        elif self.dataset == "kitti":
            projected_pix_3d = projected_pix.reshape(
                b,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                3,
            )
            projected_pix_2d = projected_pix.reshape(
                b,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale * self.scene_size[2] // self.project_scale,
                3,
            )[:, :, :, :2]
            x_3d = sum([F.grid_sample(_, projected_pix_3d, align_corners=True) for _ in x if _.ndim == 5])
            x_2d = sum([F.grid_sample(_, projected_pix_2d, align_corners=True).reshape(*x_3d.shape) for _ in x if _.ndim == 4])
            x = x_2d + x_3d

        return x
