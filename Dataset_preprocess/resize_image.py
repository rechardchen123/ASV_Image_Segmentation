#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:59
# @File  : resize_image.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import os

'''
Resize images to your designated size. 
'''

image_width = 512
image_height = 384
source_path = '/home/richardchen123/Desktop/asv_image_segmentation/test_images/test_MODD2/images/'
target_path = '/home/richardchen123/Desktop/asv_image_segmentation/test_images/test_MODD2/revised_images/'

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)

i = 0
for file in image_list:
    i = i +1
    image_source = cv2.imread(os.path.join(source_path, file))
    image = cv2.resize(image_source, (image_width, image_height), cv2.INTER_LINEAR)
    cv2.imwrite(target_path + str(i)+'.jpg', image)
print('Done!')