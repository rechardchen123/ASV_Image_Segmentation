#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:48
# @File  : cfg.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
BATCH_SIZE = 2
NUM_CLASSES = 3
EPOCH_NUMBER = 100
TRAIN_ROOT = './dataset/train'
TRAIN_LABEL = './dataset/train_labels'
VAL_ROOT = './dataset/val'
VAL_LABEL = './dataset/val_labels'
TEST_ROOT = './dataset/test'
TEST_LABEL = './dataset/test_labels'
class_dict_path = './dataset/class_dict.csv'
SEQ_TXT = './dataset/frame_name.txt'
SAVE_DIR = './result_pics/'
MODEL_WEIGHTS = './WODIS_weights.pth'  # for inference
RESUME = True
crop_size = (384,512) # image height and width for preprocessing
#IMG_SIZE = (480, 640)
IMG_SIZE = (384, 512)  # the input image size for the training, inference
