import torch
import torch.nn as nn
import torch.nn.functional as F


class FLoSP(nn.Module):
    def __init__(self, scene_size, dataset, project_scale, feature):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale
        self.conv = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(feature, momentum=0.001),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, projected_pix, fov_mask):
        b, c, d, h, w = x[0].shape
        if self.dataset == "NYU":
            projected_pix[:,:,2] *= 127 # [0-479, 0-639, 0-127]
            x_shape = torch.FloatTensor([[19, 14, 15], [39, 29, 31], [79, 59, 63], [159, 119, 127]]).to(projected_pix).reshape(4, 1, 1, 1, 1, 3)
            x_stride = torch.FloatTensor([[32, 32, 8], [16, 16, 4], [8, 8, 2], [4, 4, 1]]).to(projected_pix).reshape(4, 1, 1, 1, 1, 3)
            projected_pix = projected_pix.reshape(
                b,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                3,
            )
            fov_mask = fov_mask.reshape(
                b,
                1,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                self.scene_size[1] // self.project_scale,
            )
            x = sum([F.grid_sample(_, projected_pix / (st * sh / 2) - 1, padding_mode='border', align_corners=True) for _, sh, st in zip(x, x_shape, x_stride)])
            x.masked_fill_(~fov_mask.expand(-1, x.shape[1], -1, -1, -1), 0)
            x = x.permute(0, 1, 2, 4, 3)
        elif self.dataset == "kitti":
            projected_pix[:, :, 2] *= 127 # [0-479, 0-639, 0-127]
            x_shape = torch.FloatTensor([[38, 11, 15], [76, 23, 31], [152, 46, 63], [304, 92, 127]]).to(projected_pix).reshape(4, 1, 1, 1, 1, 3)
            x_stride = torch.FloatTensor([[32, 32, 8], [16, 16, 4], [8, 8, 2], [4, 4, 1]]).to(projected_pix).reshape(4, 1, 1, 1, 1, 3)
            projected_pix = projected_pix.reshape(
                b,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                3,
            )
            fov_mask = fov_mask.reshape(
                b,
                1,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
            )
            x = sum([F.grid_sample(_, projected_pix / (st * sh / 2) - 1, padding_mode='border', align_corners=True) for _, sh, st in zip(x, x_shape, x_stride)])
            x.masked_fill_(~fov_mask.expand(-1, x.shape[1], -1, -1, -1), 0)
            
        return self.conv(x)
