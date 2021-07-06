#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:48
# @File  : dataset.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg

'''
Dataset process:
1. for all the masks, it firstly needs to be coded with the method called hash algorithm. 
The method called 'encode_label_pix' combines the color and label correspondence. The following uses a method similar to 
256 hexadecimal to map each pixel in the color map to the category it represents.
- Hash function: (cm[0] * 256 + cm[1]) * 256 + cm[2]
- Hash map: cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
- Hash Table: cm2lbl
Example: one pixel value P(128, 64, 128), decoded with Hash function (P[0] * 256 + P[1]) * 256 + P[2] to a integer value
（8405120). The number 8405120 as a pixel P in the Hash Table to index cm2lbl[8405120] to find the pixel 
value P(128, 64, 128) category. 

2. The image preprocessing: transform, image resize etc in class ASV_ImageDataSet. After preprocessing, it will return
a dictionary 'sample = {'img': img, 'label': label}'. The data will be fed into the model for training or testing.
'''


class LabelProcessor:
    '''encoding for the each image label'''

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
        read the dataset and preprocessing
        :param file_path: the data and label file location, respectively
        :param crop_zie: cropping image size [width, height]
        '''
        # 1. read the image and label file location
        if len(file_path) != 2:
            raise ValueError('read the image and label file location...')
        self.img_path = file_path[0]
        self.label_path = file_path[1]

        # 2. get the image and mask names
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)

        # 3. initialise the settings
        self.crop_size = crop_size

    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]
        # read the image from the image list.
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transform(img, label)
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """center cropping"""
        data = F.center_crop(data, crop_size)
        label = F.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        ''' preprocessing the image and labels'''
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
