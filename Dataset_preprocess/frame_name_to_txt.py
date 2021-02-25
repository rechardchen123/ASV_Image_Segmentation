#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:57
# @File  : frame_name_to_txt.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import os

'''
In order to train the model, we need to save all the image and label name into one txt file.
'''
names= os.listdir('/home/richardchen123/Desktop/VIS_Onboard/video_frames/video_frame_11/frames')
names.sort(key=lambda x:int(x[:-4]))
print(names[:9])
image_ids = open('/home/richardchen123/Desktop/VIS_Onboard/video_frames/video_frame_11/frame_name.txt', 'w')
for name in names:
    image_ids.write('%s\n' % (name))

image_ids.close()

