#!/usr/bin/env python3
# -*- coding: utf-8 -*
# *************************************************************
# @Time  : 25/02/2021 21:47
# @File  : WODIS.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# water obstacle detection based on image scene (WODIS) model
# *************************************************************
from torchvision import models
import torch
import torch.nn as nn
from thop import profile


class baseline(nn.Module):
    """
    The base feature extraction model here and the model loaded from the Pytorch repositories.
    """

    def __init__(self, pretrainined=True):
        super(baseline, self).__init__()
        self.features = models.resnet101(pretrained=pretrainined)
        self.features.load_state_dict(torch.load('./xception_model.pth'))  # the resent-101 weight directory
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        return feature1, feature2, feature3, feature4


class AttentionRefinementModule(nn.Module):
    """
    Attention Refinement Module
    """

    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size())
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module
    """

    def __init__(self, num_classes, in_channels):
        super(FeatureFusionModule, self).__init__()
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


class WODIS_model(nn.Module):
    def __init__(self, is_training, num_classes):
        '''
        Network definition
        :param is_training: whether to update the running mean and variance of the batch normalisation layer.
                            If the batch size is small, it is better to keep the running mean and variance of the pre-trained
                            model frozen.
        :param num_classes: number of classes to predict(including background).
        '''
        super(WODIS_model, self).__init__()
        self.encoder = baseline(pretrainined=False)
        self.arm1 = AttentionRefinementModule(2048, 2048)
        self.arm2 = AttentionRefinementModule(512, 512)
        self.ffm = FeatureFusionModule(num_classes, 3072)
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        output1, output2, output3, output4 = self.encoder(input)

        arm1_out = self.arm1(output4)
        arm1_out = torch.nn.functional.interpolate(arm1_out, size=output2.size()[-2:], mode='bilinear')
        arm2_out = self.arm2(output2)
        arm12_out = torch.cat((arm1_out, arm2_out), dim=1)

        ffm_out = self.ffm(arm12_out, output2)
        result = torch.nn.functional.interpolate(ffm_out, size=input.size()[-2:], mode='bilinear')
        result = self.conv(result)
        if self.training == True:
            return result, output3
        return result


if __name__ == '__main__':
    import torch as t

    rgb = t.randn(1, 3, 352, 480)
    net = WODIS_model(is_training=True, num_classes=3).eval()
    out = net(rgb)
    flops, params = profile(net, (rgb,))
    print('flops: ', flops / (1000 ** 3), 'params: ', params / (1000 ** 2))
