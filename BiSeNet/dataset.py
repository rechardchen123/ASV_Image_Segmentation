#!/usr/bin/env python3
# -*- coding: utf-8 -*
# **************************************************
# @Time  : 18/12/2020 14:53
# @File  : bisenet.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# **************************************************
import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg


class LabelProcessor:
    '''对标签进行编码'''

    def __init__(self, file_path):
        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class ASV_ImageDataSet(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        '''
        读取数据集并处理
        :param file_path: 数据和标签的路径，第一个元素表示图片路径，第二个为标签路径
        :param crop_zie: 裁剪的图像的大小
        '''
        # 1. 正确读入图片和标签的路径
        if len(file_path) != 2:
            raise ValueError('同时需要读入图片和标签文件夹的路径，图片的路径在前！')
        self.img_path = file_path[0]
        self.label_path = file_path[1]

        # 2. 从路径中取出图片和对应的mask的文件名保存在两个列表中
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)

        # 3. 初始化数据处理函数的设置
        self.crop_size = crop_size

    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]
        # 从文件名中读取数据
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transform(img, label)
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        '''从文件夹中读取数据'''
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        '''裁剪输入的图像和标签大小'''
        data = F.center_crop(data, crop_size)
        label = F.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        '''对图片和标签数值处理'''
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)
        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)

if __name__ == "__main__":
    TRAIN_ROOT = './dataset/train'
    TRAIN_LABEL = './dataset/train_labels'
    # VAL_ROOT = './dataset/val'
    # VAL_LABEL = './dataset/val_labels'
    # TEST_ROOT = './dataset/test'
    # TEST_LABEL = './dataset/test_labels'
    crop_size = (512, 384)
    ASV_train = ASV_ImageDataSet([TRAIN_ROOT, TRAIN_LABEL], crop_size)

