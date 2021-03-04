#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:49
# @File  : inference.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import os
import pandas as pd
import time
import numpy as np
import torch as t
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from WODIS import WODIS_model
import cv2
import cfg


def main():
    if t.cuda.is_available():
        device = t.device('cuda')
    else:
        device = t.device('cpu')

    '''
    create the model and start the inference...
    '''
    # load model
    net = WODIS_model(is_training=False, num_classes=cfg.NUM_CLASSES).to(device)
    checkpoint = t.load(cfg.MODEL_WEIGHTS, map_location='cpu')
    net.load_state_dict(checkpoint)
    pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []

    # read label color from the class dict
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    # create output folder if it does not exist.
    if not os.path.exists(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)

    # get number of lines in text file
    num_imgs = sum(1 for line in open(cfg.SEQ_TXT))

    # perform inferences on dataset
    f_id = open(cfg.SEQ_TXT, 'r')

    counter = 1
    sum_times = 0

    for line in f_id:
        image_name = line.strip('\n')
        image_base_name = image_name.split('.')[0]
        # read image
        img_path = os.path.join(cfg.TEST_ROOT, image_name)
        img = cv2.imread(img_path)
        if img is None:
            break
        img_reverse = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.Resize(cfg.IMG_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # read one image for the inference process, it needs to expand the first dimension to match the tensor.
        img_out = transform(img_reverse).unsqueeze(0)

        # inference starting...
        start_time = time.time()
        valImg = img_out.to(device)
        out, cx1 = net(valImg)
        elapsed_time = time.time() - start_time
        sum_times += elapsed_time
        print('Elapsed time: %.04f for image num %03d' % (elapsed_time, counter))
        out = F.log_softmax(out, dim=1)
        pre_label = out.max(dim=1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        cv2.imwrite(cfg.SAVE_DIR + image_base_name + '.png', pre)
        counter += 1
    f_id.close()
    print('Average time per image: %.5f' % (sum_times / num_imgs))


if __name__ == '__main__':
    main()
