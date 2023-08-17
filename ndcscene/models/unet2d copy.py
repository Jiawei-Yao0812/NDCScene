
"""
Code adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .resnet import build_model

class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        expansion,
        stride=1,
        dilation=[1, 1, 1],
        downsample=None,
        bn_momentum=0.0003,
    ):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=True)
        # self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=(dilation[0], dilation[1], dilation[2]),
            padding=(dilation[0], dilation[1], dilation[2]),
            bias=True,
        )
        # self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes * expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm3d(planes * expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.relu(x + residual)

        return x

class Process(nn.Module):
    def __init__(self, feature, bn_momentum, dilations=[1, 2, 3], reduction=4):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[
                Bottleneck3D(
                    feature,
                    feature // reduction,
                    reduction,
                    dilation=[i, i, i],
                    bn_momentum=bn_momentum,
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)

class UpSampleBN(nn.Module):
    def __init__(self, input_features, skip_features, output_features, bn_momentum):
        super(UpSampleBN, self).__init__()
        self.conv_1 = nn.ConvTranspose3d(input_features, output_features, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_2 = nn.Conv2d(skip_features, output_features, kernel_size=3, stride=1, padding=1)
        self.net = nn.Sequential(
            nn.BatchNorm3d(output_features, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, concat_with):
        return self.net(self.conv_1(x).add_(self.conv_2(concat_with).unsqueeze(2)))

class DUpSampleBN(nn.Module):
    def __init__(self, input_features, output_features, bn_momentum):
        super(DUpSampleBN, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(input_features, output_features, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), output_padding=(1, 0, 0)),
            nn.BatchNorm3d(output_features, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class XYUpSampleBN(nn.Module):
    def __init__(self, input_features, skip_features, output_features, bn_momentum):
        super(XYUpSampleBN, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(input_features, output_features, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_2 = nn.Conv2d(skip_features, output_features, kernel_size=3, stride=1, padding=1)
        self.net = nn.Sequential(
            nn.BatchNorm3d(output_features, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, concat_with):
        return self.net(self.conv_1(x).add_(self.conv_2(concat_with)))

class DecoderBN(nn.Module):
    def __init__(
        self, bottleneck_features, num_features, out_feature, with_2d=False,
    ):
        super(DecoderBN, self).__init__()
        self.with_2d = with_2d
        feature_1 = out_feature * 4 # 800
        feature_2 = out_feature * 2 # 400
        feature_3 = out_feature # 200

        self.conv_1 = nn.Conv2d(bottleneck_features[-1], feature_1 * 2, kernel_size=3, stride=1, padding=1) # 800
        self.conv_2 = nn.Sequential(nn.Conv3d(feature_1, feature_3, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)) # 800
        self.conv_3 = nn.Sequential(nn.Conv3d(feature_2, feature_3, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)) # 800
        self.up_1 = DUpSampleBN(feature_1, feature_1, bn_momentum=0.001)
        self.bl_1 = Bottleneck3D(feature_1, feature_1 // 4, 4, stride=1, dilation=[1, 1, 1], downsample=None, bn_momentum=0.001)
        self.up_2 = DUpSampleBN(feature_1, feature_1, bn_momentum=0.001)
        self.bl_2 = Bottleneck3D(feature_1, feature_1 // 4, 4, stride=1, dilation=[1, 1, 1], downsample=None, bn_momentum=0.001)
        self.up_3 = DUpSampleBN(feature_1, feature_1, bn_momentum=0.001)
        self.bl_3 = Bottleneck3D(feature_1, feature_1 // 4, 4, stride=1, dilation=[1, 1, 1], downsample=None, bn_momentum=0.001)
        self.up_4 = UpSampleBN(input_features=feature_1, skip_features=bottleneck_features[-2], output_features=feature_2, bn_momentum=0.001)
        self.bl_4 = Process(feature_2, bn_momentum=0.001)
        self.up_5 = UpSampleBN(input_features=feature_2, skip_features=bottleneck_features[-3], output_features=feature_3, bn_momentum=0.001)
        self.bl_5 = Process(feature_3, bn_momentum=0.001)
        if self.with_2d:
            self.up_1_xy = XYUpSampleBN(bottleneck_features[-1], bottleneck_features[-2], out_feature, bn_momentum=0.001)
            self.up_2_xy = XYUpSampleBN(out_feature, bottleneck_features[-3], out_feature, bn_momentum=0.001)
            self.up_3_xy = XYUpSampleBN(out_feature, bottleneck_features[-4], out_feature, bn_momentum=0.001)
            self.up_4_xy = XYUpSampleBN(out_feature, bottleneck_features[-5], out_feature, bn_momentum=0.001)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features
        b, _, h, w = x_block4.shape

        x_1 = self.conv_1(x_block4).reshape(b, 2, -1, h, w).permute(0, 2, 1, 3, 4) # [B, C, 2, 15, 20]
        x_2 = self.bl_1(self.up_1(x_1)) # [B, C, 4, 15, 20]
        x_3 = self.bl_2(self.up_2(x_2)) # [B, C, 8, 15, 20]
        x_4 = self.bl_3(self.up_3(x_3)) # [B, C, 16, 15, 20]
        x_5 = self.bl_4(self.up_4(x_4, x_block3)) # [B, C, 32, 30, 40]
        x_6 = self.bl_5(self.up_5(x_5, x_block2)) # [B, C, 64, 60, 80]
        res = [self.conv_2(x_4), self.conv_3(x_5), x_6]

        if self.with_2d:
            x_2_xy = self.up_1_xy(x_block4, x_block3)
            x_3_xy = self.up_2_xy(x_2_xy, x_block2)
            x_4_xy = self.up_3_xy(x_3_xy, x_block1)
            x_5_xy = self.up_4_xy(x_4_xy, x_block0)
            res.append(x_5_xy)
        return res

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        self.apply(set_bn_eval)
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return [features[4], features[5], features[6], features[8], features[11]]


class UNet2D(nn.Module):
    def __init__(self, backend_name, backend, bottleneck_features, num_features, out_feature, with_2d=False):
        super(UNet2D, self).__init__()
        self.encoder = Encoder(backend) if 'efficientnet' in backend_name else backend
        self.decoder = DecoderBN(
            bottleneck_features=bottleneck_features,
            num_features=num_features,
            out_feature=out_feature,
            with_2d=with_2d,
        )

    def forward(self, x, **kwargs):
        encoded_feats = self.encoder(x)
        unet_out = self.decoder(encoded_feats, **kwargs)
        return unet_out

    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    @classmethod
    def build(cls, basemodel_name, **kwargs):
        # basemodel_name = "tf_efficientnet_b7_ns"
        bottleneck_features = {
                                "tf_efficientnet_b7_ns": (32, 48, 80, 224, 2560),
                                "RN50": (512, 1024, 2048),
                                "RN101": (512, 1024, 2048),
                                "RN50x4": (640, 1280, 2560),
                                "RN50x16": (768, 1536, 3072),
                                }[basemodel_name]
        num_features = 2048

        if basemodel_name == "tf_efficientnet_b7_ns":
            print("Loading base model ()...".format(basemodel_name), end="")
            basemodel = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
            )
            print("Done.")

            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
        else:
            basemodel = build_model(basemodel_name)

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel_name, basemodel, bottleneck_features=bottleneck_features, num_features=num_features, **kwargs)
        print("Done.")
        return m

if __name__ == '__main__':
    model = UNet2D.build("tf_efficientnet_b7_ns", out_feature=256).cuda()
    x = torch.zeros(2, 3, 480, 640).cuda()
    y = model(x)
    z = model(x)
    print(y.shape)
    print(torch.cuda.memory_allocated() / 4 / 1024 / 1024 / 1024)
