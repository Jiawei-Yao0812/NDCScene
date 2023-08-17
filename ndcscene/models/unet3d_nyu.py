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


# class Process(nn.Module):
#     def __init__(
#         self,
#         feature,
#         norm_layer,
#         bn_momentum=0.001,
#     ):
#         super(Process, self).__init__()
#         self.conv1 = nn.Conv3d(
#             feature,
#             feature // 2,
#             kernel_size=3,
#             dilation=1,
#             padding=1,
#             bias=True,
#         )
#         self.conv2 = nn.Conv3d(feature // 2, feature, kernel_size=1, bias=False)
#         self.bn2 = norm_layer(feature, momentum=bn_momentum)
#         self.conv3 = nn.Conv3d(
#             feature,
#             feature // 2,
#             kernel_size=3,
#             dilation=3,
#             padding=3,
#             bias=True,
#         )
#         self.conv4 = nn.Conv3d(feature // 2, feature, kernel_size=1, bias=False)
#         self.bn4 = norm_layer(feature, momentum=bn_momentum)

#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x = self.relu(x + self.bn2(self.conv2(self.relu(self.conv1(x)))))
#         x = self.relu(x + self.bn4(self.conv4(self.relu(self.conv3(x)))))

#         return x

class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
        full_scene_size,
        n_relations=4,
        context_prior=True,
        bn_momentum=0.001,
    ):
        super(UNet3D, self).__init__()

        self.feature_1_4 = feature
        self.feature_1_8 = feature * 2
        self.feature_1_16 = feature * 4

        self.feature_1_16_dec = self.feature_1_16
        self.feature_1_8_dec = self.feature_1_8
        self.feature_1_4_dec = self.feature_1_4

        self.process_1_4 = nn.Sequential(
            Downsample(self.feature_1_4, norm_layer, bn_momentum),
            Process(self.feature_1_8, norm_layer, bn_momentum),
        )
        self.process_1_8 = nn.Sequential(
            Downsample(self.feature_1_8, norm_layer, bn_momentum),
            Process(self.feature_1_16, norm_layer, bn_momentum),
        )
        # self.process_1_4 = Downsample(self.feature_1_4, norm_layer, bn_momentum)
        # self.process_1_8 = Downsample(self.feature_1_8, norm_layer, bn_momentum)
        self.up_1_16_1_8 = Upsample(
            self.feature_1_16_dec, self.feature_1_8_dec, norm_layer, bn_momentum
        )
        self.up_1_8_1_4 = Upsample(
            self.feature_1_8_dec, self.feature_1_4_dec, norm_layer, bn_momentum
        )
        self.ssc_head_1_4 = SegmentationHead(
            self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3]
        )

        self.context_prior = context_prior
        size_1_16 = tuple(np.ceil(i / 4).astype(int) for i in full_scene_size)

        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature_1_16,                
                size_1_16,
                n_relations=n_relations,
                bn_momentum=bn_momentum,
            )

    #
    def forward(self, input_dict):
        res = {}

        x3d_1_4 = input_dict["x3d"]
        x3d_1_8 = self.process_1_4(x3d_1_4)
        x3d_1_16 = self.process_1_8(x3d_1_8)

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_1_16)
            x3d_1_16 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_1_8 = self.up_1_16_1_8(x3d_1_16) + x3d_1_8
        x3d_up_1_4 = self.up_1_8_1_4(x3d_up_1_8) + x3d_1_4

        ssc_logit_1_4 = self.ssc_head_1_4(x3d_up_1_4)

        res["ssc_logit"] = ssc_logit_1_4

        return res

if __name__ == '__main__':
    model = UNet3D(
                        class_num=12,
                        norm_layer=nn.BatchNorm3d,
                        feature=200,
                        full_scene_size=[60, 36, 60],
                        n_relations=4,
                        context_prior=True,
                        bn_momentum=0.1
        ).cuda()
    x = torch.zeros(2, 200, 60, 36, 60).cuda()
    res = model({'x3d': x})
    print(res["ssc_logit"].shape)
    print(torch.cuda.memory_allocated() / 4 / 1024 / 1024 / 1024)