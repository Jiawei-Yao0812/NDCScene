# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ndcscene.models.CRP3D import CPMegaVoxels
from ndcscene.models.modules import (
    Process,
    Upsample,
    Downsample,
    SegmentationHead,
    LightSegmentationHead,
)


class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        context_prior=None,
        bn_momentum=0.001,
    ):
        super(UNet3D, self).__init__()
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature

        size_l1 = (
            int(self.full_scene_size[0] / project_scale),
            int(self.full_scene_size[1] / project_scale),
            int(self.full_scene_size[2] / project_scale),
        )
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        dilations = [1, 2, 3]
        self.process_l1 = nn.Sequential(
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Downsample(self.feature * 2, norm_layer, bn_momentum),
        )
        self.process_l3 = nn.Sequential(
            Downsample(self.feature * 4, norm_layer, bn_momentum),
        )

        self.up_14_l3 = Upsample(
            self.feature * 8, self.feature * 4, norm_layer, bn_momentum
        )
        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.up_l1_lfull = Upsample(
            self.feature, self.feature // 2, norm_layer, bn_momentum
        )

        self.ssc_head = LightSegmentationHead(
            self.feature // 2, class_num, dilations
        )

        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature * 4, size_l3, bn_momentum=bn_momentum
            )

    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"]

        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        x3d_l4 = self.process_l3(x3d_l3)

        x3d_up_l3 = self.up_14_l3(x3d_l4)    + x3d_l3

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_up_l3)
            x3d_up_l3 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_l2 = self.up_13_l2(x3d_up_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        ssc_logit_full = self.ssc_head(x3d_up_lfull)

        res["ssc_logit"] = ssc_logit_full

        return res

    # def forward(self, input_dict):
    #     res = {}

    #     x3d_l1 = input_dict["x3d"]

    #     x3d_l2 = self.process_l1(x3d_l1)

    #     x3d_l3 = self.process_l2(x3d_l2)

    #     if self.context_prior:
    #         ret = self.CP_mega_voxels(x3d_l3)
    #         x3d_l3 = ret["x"]
    #         for k in ret.keys():
    #             res[k] = ret[k]

    #     x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
    #     x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
    #     x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

    #     ssc_logit_full = self.ssc_head(x3d_up_lfull)

    #     res["ssc_logit"] = ssc_logit_full

    #     return res
