import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

import sys, os


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=6, stride=1, dilation=1):
        super(InvertedResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = expansion
        self.s = stride
        self.dilation = dilation
        self.inverted_residual_block()

    def inverted_residual_block(self):

        block = []

        # 1x1 Convolution / Bottleneck
        block.append(nn.Conv2d(self.in_channels, self.in_channels*self.t, kernel_size=1, bias=False))
        block.append(nn.BatchNorm2d(self.in_channels*self.t))
        block.append(nn.ReLU6(inplace=True))

        # 3x3 Depthwise Convolution
        block.append(nn.Conv2d(self.in_channels*self.t, self.in_channels*self.t, kernel_size=3, stride=self.s, padding=self.dilation, 
                                dilation=self.dilation, groups = self.in_channels*self.t, bias=False ))
        block.append(nn.BatchNorm2d(self.in_channels*self.t))
        block.append(nn.ReLU6(inplace=True))

        # Linear 1x1 Convolution
        block.append(nn.Conv2d(self.in_channels*self.t, self.out_channels, kernel_size=1, stride=self.s, bias=False))
        block.append(nn.BatchNorm2d(self.out_channels))

        self.block = nn.Sequential(*block)


        if self.in_channels != self.out_channels and self.s != 2:
            self.res_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                                          nn.BatchNorm2d(self.out_channels))
        else:
            self.res_conv = None

    def forward(self, x):
        if self.s == 1:
            # Use residual connection
            if self.res_conv is None:
                out = x + self.block(x)
            else:
                out = self.res_conv(x) + self.block(x)

        else:
            out = self.block(x)

        return out


def get_inverted_residual_block_arr(in_, out_, t=6, s=1, n=1):
    block = []
    block.append(InvertedResidual(in_, out_, t, s))
    for i in range(n-1):
        block.append(InvertedResidual(out_, out_, t, 1))
    return block



class ASPPModule(nn.Module):
    """

    The ASPP module implemented by Speedinghzl in the pytorch-segmentation-toolbox Github Repository.
    The link can be found here. https://github.com/speedinghzl/pytorch-segmentation-toolbox

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


class MobileNetv2_DeepLabv3(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetv2_DeepLabv3, self).__init__()

        self.num_classes = num_classes

        self.s = [2, 1, 2, 2, 2, 1, 1, 1]  # stride of each conv stage
        self.t = [1, 1, 6, 6, 6, 6, 6, 6]  # expansion factor t
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]  # number of repeat time
        self.c = [32, 16, 24, 32, 64, 96, 160, 320]  # output channel of each conv stage

        block = []

        # MobileNetV2 first layer
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=self.c[0], kernel_size=3, stride=self.s[0], padding=1, bias=False),
                                       nn.BatchNorm2d(self.c[0]),
                                       # nn.Dropout2d(self.dropout_prob, inplace=True),
                                       nn.ReLU6(inplace=True))

        # MobileNetV2 second to seventh layer
        for i in range(7):
                block.extend(get_inverted_residual_block_arr(self.c[i], self.c[i+1],
                                                                t=self.t[i+1], s=self.s[i+1],
                                                                n=self.n[i+1]))
        self.layer2to7 = nn.Sequential(*block)

        block.clear()

        # Atrous convolution layers follows the structure of MobileNet 7th Layer parameters
        # The multigrid used here assumed to be (1,2,4) where the dilation rates = 2 which produces (2,4,8)
        block.append(InvertedResidual(self.c[-1], self.c[-1], expansion=self.t[-1], stride=1, dilation=2))
        block.append(InvertedResidual(self.c[-1], self.c[-1], expansion=self.t[-1], stride=1, dilation=4))
        block.append(InvertedResidual(self.c[-1], self.c[-1], expansion=self.t[-1], stride=1, dilation=8))
        
        self.layer8 = nn.Sequential(*block)

        # # Atrous Spatial Pyramid Pooling Module
        self.head = nn.Sequential(ASPPModule(320),
                                    nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.initialize()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2to7(x)
        x = self.layer8(x)
        x = self.head(x)
        return x

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = MobileNetv2_DeepLabv3(3)
    print(model)

