#!/usr/bin/env python3
# -*- coding: utf-8 -*
# **************************************************
# @Time  : 18/12/2020 14:53
# @File  : bisenet.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# **************************************************
BATCH_SIZE = 2
EPOCH_NUMBER = 50
TRAIN_ROOT = './dataset/train'
TRAIN_LABEL = './dataset/train_labels'
VAL_ROOT = './dataset/val'
VAL_LABEL = './dataset/val_labels'
TEST_ROOT = './dataset/test'
TEST_LABEL = './dataset/test_labels'
class_dict_path = './dataset/class_dict.csv'
crop_size = (384, 512)  #高度和宽度 顺序很重要
