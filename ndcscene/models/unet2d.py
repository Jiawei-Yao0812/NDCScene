
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
    def __init__(self, input_features_3d, input_features_2d, skip_features, output_features, output_padding, bn_momentum):
        super(UpSampleBN, self).__init__()
        self.deconv_3d = nn.ConvTranspose3d(input_features_3d, output_features, kernel_size=3, stride=2, padding=1, output_padding=(1, output_padding[0], output_padding[1]))
        self.deconv_2d = nn.ConvTranspose2d(input_features_2d, output_features, kernel_size=3, stride=2, padding=1, output_padding=output_padding)
        self.conv_2d = nn.Conv2d(skip_features, output_features, kernel_size=3, stride=1, padding=1)
        self.net_3d = nn.Sequential(
            nn.BatchNorm3d(output_features, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
        )
        self.net_2d_1 = nn.Sequential(
            nn.BatchNorm2d(output_features, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        )
        self.net_2d_2 = nn.Sequential(
            nn.BatchNorm2d(output_features, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x_3d, x_2d, concat_with):
        y_2d = self.net_2d_1(self.deconv_2d(x_2d).add_(self.conv_2d(concat_with)))
        y_3d = self.net_3d(self.deconv_3d(x_3d).add_(y_2d.unsqueeze(2)))
        y_2d = self.net_2d_2(y_2d)
        return y_3d, y_2d

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

class DecoderBN(nn.Module):
    def __init__(self, bottleneck_features, num_features, out_feature, output_padding):
        super(DecoderBN, self).__init__()
        feature_1 = out_feature * 4 # 800
        feature_2 = out_feature * 2 # 400
        feature_3 = out_feature # 200

        self.pr_1 = nn.Conv2d(bottleneck_features[-1], feature_1 * 2, kernel_size=3, stride=1, padding=1) # 800
        self.up_1 = DUpSampleBN(feature_1, feature_1, bn_momentum=0.001)
        self.bl_1 = Bottleneck3D(feature_1, feature_1 // 4, 4, stride=1, dilation=[1, 1, 1], downsample=None, bn_momentum=0.001)
        self.up_2 = DUpSampleBN(feature_1, feature_1, bn_momentum=0.001)
        self.bl_2 = Bottleneck3D(feature_1, feature_1 // 4, 4, stride=1, dilation=[1, 1, 1], downsample=None, bn_momentum=0.001)
        self.up_3 = DUpSampleBN(feature_1, feature_1, bn_momentum=0.001)
        self.bl_3 = Bottleneck3D(feature_1, feature_1 // 4, 4, stride=1, dilation=[1, 1, 1], downsample=None, bn_momentum=0.001)
        self.up_4 = UpSampleBN(input_features_3d=feature_1, input_features_2d=bottleneck_features[-1], skip_features=bottleneck_features[-2], output_features=feature_2, output_padding=output_padding[0], bn_momentum=0.001)
        self.bl_4 = Process(feature_2, bn_momentum=0.001)
        self.up_5 = UpSampleBN(input_features_3d=feature_2, input_features_2d=feature_2, skip_features=bottleneck_features[-3], output_features=feature_3, output_padding=output_padding[1], bn_momentum=0.001)
        self.bl_5 = Process(feature_3, bn_momentum=0.001)
        self.up_6 = UpSampleBN(input_features_3d=feature_3, input_features_2d=feature_3, skip_features=bottleneck_features[-4], output_features=feature_3, output_padding=output_padding[2], bn_momentum=0.001)
        self.lt_4 = nn.Conv3d(feature_1, out_feature, kernel_size=1, stride=1, padding=0)
        self.lt_5 = nn.Conv3d(feature_2, out_feature, kernel_size=1, stride=1, padding=0)
        self.lt_6 = nn.Conv3d(feature_3, out_feature, kernel_size=1, stride=1, padding=0)
        self.lt_7 = nn.Conv3d(feature_3, out_feature, kernel_size=1, stride=1, padding=0)


    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features
        b, _, h, w = x_block4.shape

        x_1_3d = self.pr_1(x_block4).reshape(b, 2, -1, h, w).permute(0, 2, 1, 3, 4) # [B, C, 2, 15, 20]
        x_2_3d = self.bl_1(self.up_1(x_1_3d)) # [B, C, 4, 15, 20]
        x_3_3d = self.bl_2(self.up_2(x_2_3d)) # [B, C, 8, 15, 20]
        x_4_3d = self.bl_3(self.up_3(x_3_3d)) # [B, C, 16, 15, 20]
        x_5_3d, x_5_2d = self.up_4(x_4_3d, x_block4, x_block3) # [B, C, 32, 30, 40]
        x_5_3d = self.bl_4(x_5_3d)
        x_6_3d, x_6_2d = self.up_5(x_5_3d, x_5_2d, x_block2) # [B, C, 64, 60, 80]
        x_6_3d = self.bl_5(x_6_3d)
        x_7_3d, x_7_2d = self.up_6(x_6_3d, x_6_2d, x_block1) # [B, C, 128, 120, 160]

        res = [self.lt_4(x_4_3d), self.lt_5(x_5_3d), self.lt_6(x_6_3d), self.lt_7(x_7_3d)]

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
    def __init__(self, backend_name, backend, bottleneck_features, num_features, out_feature, output_padding):
        super(UNet2D, self).__init__()
        self.encoder = Encoder(backend) if 'efficientnet' in backend_name else backend
        self.decoder = DecoderBN(
            bottleneck_features=bottleneck_features,
            num_features=num_features,
            out_feature=out_feature,
            output_padding=output_padding,
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
