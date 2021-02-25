#!/usr/bin/env python3
# -*- coding: utf-8 -*
# **************************************************
# @Time  : 18/12/2020 14:53
# @File  : bisenet.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# **************************************************

import os
import numpy as np
from PIL import Image

def create_visual_anno(anno):
    label2color_dict = {
        0:[128,0,0],   # 标签0代表的是障碍物
        1:[192,128,0],   # 标签1代表的是水面
        2:[128,128,128]    # 标签2代表的是天空
    }
    visual_anno = np.zeros((anno.shape[0], anno.shape[1],3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i,j]]
            visual_anno[i,j,0] = color[0]
            visual_anno[i,j,1] = color[1]
            visual_anno[i,j,2] = color[2]
    return visual_anno



if __name__ == "__main__":
    MASK_PATH = './dataset/raw_dataset/masks'
    output_address = './dataset/raw_dataset/transformed_mask'
    for filename in os.listdir(MASK_PATH):
        print(filename)
        img = Image.open(MASK_PATH +"/" + filename)
        img = np.array(img)
        img_output = create_visual_anno(img)
        img = Image.fromarray(img_output.astype('uint8')).convert('RGB')
        img.save('./dataset/raw_dataset/transformed_mask'+'/'+filename+'.png')




    



