import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from layers import InvertedResidual, conv_bn
from collections import OrderedDict
from functools import partial


class MobileNetV2ASPP(nn.Module):
    def __init__(self, n_class=3, in_size=(224, 448), width_mult=1.,
                 out_sec=256, aspp_sec=(12, 24, 36)):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2ASPP, self).__init__()

        self.n_class = n_class
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],    # 1/2
            [6, 24, 2, 2, 1],    # 1/4
            [6, 32, 3, 2, 1],    # 1/8
            [6, 64, 4, 1, 2],    # 1/8
            [6, 96, 3, 1, 4],    # 1/8
            [6, 160, 3, 1, 8],   # 1/8
            [6, 320, 1, 1, 16],  # 1/8
        ]

        # building first layer
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0

        self.input_size = in_size

        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([("conv1", conv_bn(inp=3, oup=input_channel, stride=2))]))

        # building inverted residual blocks
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)

            # Create blocks for module
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=s,
                                                                                dilate=1,
                                                                                expand_ratio=t)))
                else:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=1,
                                                                                dilate=d,
                                                                                expand_ratio=t)))

                input_channel = output_channel

            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        # building last several layers
        org_last_chns = (self.interverted_residual_setting[0][1] +
                         self.interverted_residual_setting[1][1] +
                         self.interverted_residual_setting[2][1] +
                         self.interverted_residual_setting[3][1] +
                         self.interverted_residual_setting[4][1] +
                         self.interverted_residual_setting[5][1] +
                         self.interverted_residual_setting[6][1])


        self.aspp = nn.Sequential(ASPPModule(320),
                                    nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0, bias=True))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # channel_shuffle: shuffle channels in groups
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)

        return x

    def forward(self, x):

        stg1 = self.mod1(x)     # (N, 32,   224, 448)  1/2
        stg1 = self.mod2(stg1)  # (N, 16,   224, 448)  1/2 -> 1/4 -> 1/8
        stg2 = self.mod3(stg1)  # (N, 24,   112, 224)  1/4 -> 1/8
        stg3 = self.mod4(stg2)  # (N, 32,   56,  112)  1/8
        stg4 = self.mod5(stg3)  # (N, 64,   56,  112)  1/8 dilation=2
        stg5 = self.mod6(stg4)  # (N, 96,   56,  112)  1/8 dilation=4
        stg6 = self.mod7(stg5)  # (N, 160,  56,  112)  1/8 dilation=8
        stg7 = self.mod8(stg6)  # (N, 320,  56,  112)  1/8 dilation=16


        stg8 = self.aspp(stg7)
        

        return stg8



class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    # Output stride = 8 instead of 16 as the dilations are doubled. Initially they are suppose to be 6, 12, 18 for OS=16. 
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   nn.BatchNorm2d(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)

        return bottle

if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable


    model = MobileNetV2ASPP(n_class=3)
    print(model)

    # print(model.state_dict())
    # pretrained_cityscapes = torch.load("./dataset/aspp_pretrained.pth")

    # new_params = model.state_dict().copy()
    # for name, value in pretrained_cityscapes.items():
    #     new_params['.'.join(name.split('.')[1:])] = value

    # model.load_state_dict(pretrained_cityscapes, strict=False)
    # print(model.state_dict())
    # model_dict = model.state_dict()
    # # print(model_dict)
    # new_params = model.state_dict().copy

    # for name, value in pretrained_cityscapes.items():
    #     name_parts = name.split('.')
    #     newName = '.'.join(name_parts[1:])
    #     new_params[newName] = pretrained_cityscapes[value] 

    # model.load_state_dict(pretrained_cityscapes, strict=False)

    # print(model.state_dict())
    # keys = list(pretrained_cityscapes.keys())
    # keys.sort()
    # for k in keys:
    #     if "sdaspp" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod1" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod2" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod3" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod4" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod5" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod6" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod7" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "mod8" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "score" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "out_se" in k:
    #         pretrained_cityscapes.pop(k)
    #     if "last_channel" in k:
    #         pretrained_cityscapes.pop(k)
    # print(pretrained_cityscapes)

    # torch.save(model.state_dict(), "./aspp_pretrained.pth")