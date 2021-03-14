#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 12/03/2021 05:34
# @File  : heatmap.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import numpy as np
import cv2

gray_img = cv2.imread('./output_mask_1.png', flags=1)
org_img = cv2.imread('./00142.png', flags=1)

heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)

cv2.imwrite('./heatmap2.png', heat_img)

add_img = cv2.addWeighted(org_img, 0.3, heat_img, 0.7, 0)
cv2.imwrite('./add_image.png', add_img)








