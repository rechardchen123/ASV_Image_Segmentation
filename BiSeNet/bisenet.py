#!/usr/bin/env python3
# -*- coding: utf-8 -*
# **************************************************
# @Time  : 18/12/2020 14:53
# @File  : bisenet.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# **************************************************
from torchvision import models
import torch
from torch import nn
from thop import profile


class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(resnet18, self).__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.features.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = self.avgpool(feature4)
        return feature3, feature4, tail


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet_model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.saptial_path = Spatial_path()
        self.context_path = resnet18(pretrained=False)

        self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512)

        self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)

        self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)

        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        sp = self.saptial_path(input)
        print('sp shape:', sp.shape)
        cx1, cx2, tail = self.context_path(input)
        print('cx1 shape:', cx1.shape)
        print('cx2 shape:', cx2.shape)
        print('tail shape:', tail.shape)

        cx1 = self.attention_refinement_module1(cx1)
        print('cx1 shape after ARM1:', cx1.shape)
        cx2 = self.attention_refinement_module2(cx2)
        print('cx2 shape after ARM2:', cx2.shape)
        cx2 = torch.mul(cx2, tail)
        print('cx2 shape after mul:', cx2.shape)

        cx1 = torch.nn.functional.interpolate(cx1, size=sp.size()[-2:], mode='bilinear')
        print('cx1 shape after interpol:', cx1.shape)
        cx2 = torch.nn.functional.interpolate(cx2, size=sp.size()[-2:], mode='bilinear')
        print('cx2 shape after interpol:', cx2.shape)
        cx = torch.cat((cx1, cx2), dim=1)
        print('cx shape after concat:', cx.shape)

        if self.training == True:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        result = self.feature_fusion_module(sp, cx)
        print('result shape:', result.shape)

        result = torch.nn.functional.interpolate(result, size=input.size()[-2:], mode='bilinear')
        print('result shape after interpol:', result.shape)

        result = self.conv(result)
        print('final result shape:', result.shape)

        if self.training == True:
            return result, cx1_sup, cx2_sup

        return result


if __name__ == '__main__':
    import torch as t

    rgb = t.randn(1, 3, 352, 480)
    net = BiSeNet_model(19).eval()
    out = net(rgb)
    flops, params = profile(net, (rgb,))
    print('flops: ', flops / (1000 ** 3), 'params: ', params / (1000 ** 2))


