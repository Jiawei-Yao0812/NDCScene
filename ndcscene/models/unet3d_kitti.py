# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ndcscene.models.CRP3D import CPMegaVoxels
from ndcscene.models.modules import (
    Process,
    Upsample as Upsample_Ori,
    SegmentationHead,
    LightSegmentationHead,
)

class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        midplanes,
        outplanes,
        dilation=(1, 1, 1),
        stride=(1, 1, 1),
        downsample=None,
        norm_layer=nn.BatchNorm3d,
        bn_momentum=0.001,
    ):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, midplanes, kernel_size=1, bias=True)
        self.conv2 = nn.Conv3d(
            midplanes,
            midplanes,
            kernel_size=(3, 3, 3),
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=True,
        )
        self.conv3 = nn.Conv3d(
            midplanes, outplanes, kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(outplanes, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        x = self.bn3(self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))))

        if self.downsample is not None:
            residual = self.downsample(residual)
            
        x = self.relu(x + residual)

        return x

class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(
            feature,
            feature // 2,
            feature * 2,
            stride=(2, 2, 1),
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=3, stride=(2, 2, 1), padding=1),
                nn.Conv3d(
                    feature,
                    feature * 2,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                norm_layer(feature * 2, momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
            bn_momentum=bn_momentum,
        )

    def forward(self, x):
        return self.main(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=1,
                dilation=1,
                output_padding=(1, 1, 0),
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)

class Process(nn.Module):
    def __init__(
        self,
        feature,
        norm_layer,
        bn_momentum=0.001,
    ):
        super(Process, self).__init__()
        self.conv1 = nn.Conv3d(
            feature,
            feature // 2,
            kernel_size=3,
            dilation=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv3d(feature // 2, feature, kernel_size=1, bias=False)
        self.bn2 = norm_layer(feature, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            feature,
            feature // 2,
            kernel_size=3,
            dilation=3,
            padding=3,
            bias=True,
        )
        self.conv4 = nn.Conv3d(feature // 2, feature, kernel_size=1, bias=False)
        self.bn4 = norm_layer(feature, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(x + self.bn2(self.conv2(self.relu(self.conv1(x)))))
        x = self.relu(x + self.bn4(self.conv4(self.relu(self.conv3(x)))))

        return x

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
            Process(self.feature * 2, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Downsample(self.feature * 2, norm_layer, bn_momentum),
            Process(self.feature * 4, norm_layer, bn_momentum),
        )

        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.up_l1_lfull = Upsample_Ori(
            self.feature, self.feature, norm_layer, bn_momentum
        )

        self.ssc_head = LightSegmentationHead(
            self.feature, class_num, dilations
        )

        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature * 4, size_l3, bn_momentum=bn_momentum, stride_last=4,
            )

    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"]

        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_l3)
            x3d_l3 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        ssc_logit_full = self.ssc_head(x3d_up_lfull)

        res["ssc_logit"] = ssc_logit_full

        return res
