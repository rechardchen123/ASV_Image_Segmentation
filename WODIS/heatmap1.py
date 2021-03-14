#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 12/03/2021 05:58
# @File  : heatmap1.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import numpy as np

img = cv2.imread('./1.jpg', 1)
att = cv2.imread('./output_mask_1.png', 1)
w = cv2.applyColorMap(att, cv2.COLORMAP_JET)
x = img * 0.4 + w * 0.6
x = x.astype(np.uint8)
cv2.imwrite('./heatmap3.png', x)
