#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 22:00
# @File  : video_to_frames.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import os

'''
transfer the video to images
'''
video_handle = cv2.VideoCapture('/home/richardchen123/Desktop/VIS_Onboard/video_frames/video_frame_11/MVI_0804_VIS_OB.avi')
fps = int(round(video_handle.get(cv2.CAP_PROP_FPS)))
print(fps)
frame_no = 0
save_path = '/home/richardchen123/Desktop/VIS_Onboard/video_frames/video_frame_11/frames'
while True:
    eof, frame = video_handle.read()
    if not eof:
        break
    frame = cv2.resize(frame, (512, 384))
    cv2.imwrite(os.path.join(save_path, "%d.jpg" % frame_no), frame)
    frame_no += 1
    cv2.waitKey(0)