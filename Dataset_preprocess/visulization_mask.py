#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 22:01
# @File  : visulization_mask.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2

'''
Visualize the masks.
'''
imgfile = '/home/richardchen123/Desktop/ground truth1/11.jpg'   #read the raw images
pngfile = '/home/richardchen123/Desktop/ground truth1/mask_11.png'  #read the mask through the training network

img = cv2.imread(imgfile, 1)
mask = cv2.imread(pngfile, 0)
cv2.imshow('raw_image', img)   # show the raw images, no any bounding box

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]  #find the contour

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)   # draw minimum rect

cv2.imshow('mask', mask)   # show the mask
cv2.imshow('img', img)     # show the rect
cv2.imwrite('bounding_box_11.jpg', img)
cv2.waitKey(0)
