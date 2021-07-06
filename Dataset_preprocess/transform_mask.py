#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 22:07
# @File  : transform_mask.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import os
import numpy as np
from PIL import Image

'''
transform the mask from the int8 to RGB 3-channels
'''


def create_visual_anno(anno):
    label2color_dict = {
        0: [128, 0, 0],  # 0: the object
        1: [192, 128, 0],  # 1: water surface
        2: [128, 128, 128] , # 2: sky
        4: [128, 0, 0]  # void color
    }
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]
    return visual_anno


if __name__ == "__main__":
    MASK_PATH = './dataset/MaSTr1325/MaSTr1325_ground_truth_annotation'
    output_address = './dataset/MaSTr1325/transformed_annotation'
    for filename in os.listdir(MASK_PATH):
        print(filename)
        img = Image.open(MASK_PATH + "/" + filename)
        img = np.array(img)
        img_output = create_visual_anno(img)
        img = Image.fromarray(img_output.astype('uint8')).convert('RGB')
        img.save('./dataset/MaSTr1325/transformed_annotation' + '/' + filename)

