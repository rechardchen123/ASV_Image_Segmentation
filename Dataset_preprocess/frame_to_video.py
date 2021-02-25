#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:58
# @File  : frame_to_video.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import os
import cv2

'''
All inference images transfer to a video... 
'''
path = '/home/richardchen123/Desktop/VIS_Onboard/video_frames/video_frame_1/mask/'
save_path = '/home/richardchen123/Desktop/VIS_Onboard/video_frames/video_frame_1/mask.avi'
filelist = os.listdir(path)
filelist.sort(key=lambda x:int(x[-7:-4]))
print(filelist[0:8])
fps = 15
img = cv2.imread(path+'mask_001.png')
imgInfo = img.shape
size = (imgInfo[1], imgInfo[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
videoWrite = cv2.VideoWriter(save_path, fourcc, fps, size)

files = os.listdir(path)
out_num = len(files)
for i in files:
    if i.endswith('.png'):
        i = path + i
        img = cv2.imread(i)
        videoWrite.write(img)
