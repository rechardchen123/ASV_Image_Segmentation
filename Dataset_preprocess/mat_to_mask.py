#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:59
# @File  : resize_image.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

data_file = r'C:\Users\ucesxc0\Desktop\dataset\dataset_China\02\gt.mat'
data = scio.loadmat(data_file)
data.keys()
print(data.keys())
# a = data['data']

# def MatrixToImage(data):
#     data = data * 255
#     new_im = Image.fromarray(data.astype(np.uint8))
#     return new_im
#
# new_im = MatrixToImage(a)


