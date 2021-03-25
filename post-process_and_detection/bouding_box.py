#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 22:04
# @File  : bouding_box.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import numpy as np

'''
After getting all the segmentation masks, the last step is to draw the bounding boxes for the obstacles. 
'''
raw_image = cv2.imread('/home/richardchen123/Desktop/object_detection_result/MaSTr1325/1.jpg')
image = cv2.imread('/home/richardchen123/Desktop/object_detection_result/MaSTr1325/mask_1.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255,0), 2)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(image, center, radius, (0, 255,0),2)

cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
cv2.imshow('img', image)
cv2.imwrite('img_1.jpg', image)
cv2.waitKey(0)
