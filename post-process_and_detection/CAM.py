#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 22:05
# @File  : CAM.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from BiSeNet.bisenet import BiSeNet_model

'''
The class activation map (CAM) and heat map 
'''

IMG_SIZE = (480, 640)
#IMG_SIZE = (384, 512)

class Cov_features():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.data
    def remove(self):
        self.hook.remove()

def show_feature_map(img_src, conv_features):
    img = Image.open(img_file).convert('RGB')
    #height, width = img.size
    heat = conv_features.squeeze(0)
    heat_mean = torch.mean(heat, dim=0)
    heatmap = heat_mean.numpy()
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimg = heatmap*0.4 + np.array(img)[:,:,::-1]
    cv2.imwrite('./heatmap.jpg', superimg)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_file = './dataset/test/1.png'
    img = Image.open(img_file)
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=0.1307, std=0.3081)])
    img = t(img).unsqueeze(0).to(device)
    custom_model = BiSeNet_model(3).to(device)
    hook_ref = Cov_features(custom_model.conv)
    with torch.no_grad():
        custom_model(img)

    conv_features = hook_ref.features
    print('output shape', conv_features.shape)
    hook_ref.remove()