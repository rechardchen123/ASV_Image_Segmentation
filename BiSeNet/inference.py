#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 20/02/2021 18:55
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
from bisenet import BiSeNet_model
import cv2
import argparse

NUM_CLASSES = 3
SEQ_TXT = './dataset/frame_name.txt'
SAVE_DIR = './result_pics/'
DATASET_PATH = './dataset/test/'
CLASS_DICT = './dataset/class_dict.csv'
MODEL_WEIGHTS = './weights/98.pth'

# inference image size (height, width)
IMG_SIZE = (480, 640)
#IMG_SIZE = (384, 512)



def get_arguments():
    '''
    Parse all the arguments provided from the CLI.
    Returns:
        A list of parsed arguments.
    '''
    parser = argparse.ArgumentParser(description='BiseNet inference')
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH,
                        help='Path to dataset files on which inference is performed.')
    parser.add_argument('--model-weights', type=str, default=MODEL_WEIGHTS,
                        help='path to the file with model weights.')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help='Number of classes to predict.')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR,
                        help='Where to save predicted mask.')
    parser.add_argument('--seq-txt', type=str, default=SEQ_TXT,
                        help='Text sprintf to sequence txt file.')
    parser.add_argument('--class-dict', type=str, default=CLASS_DICT,
                        help='class color dictionary.')
    return parser.parse_args()


def main():
    if t.cuda.is_available():
        device = t.device('cuda')
    else:
        device = t.device('cpu')
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # create network
    net = BiSeNet_model(args.num_classes).to(device)
    checkpoint = t.load(args.model_weights, map_location='cpu')  # 如果在cpu上推理需要加上map_location的映射，在GPU上不需要
    net.load_state_dict(checkpoint)
    pd_label_color = pd.read_csv(args.class_dict, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []

    # read the label color from the class dict
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    # create output folder if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Get number of lines in text file
    num_imgs = sum(1 for line in open(args.seq_txt))

    # perform inferences on dataset
    f_id = open(args.seq_txt, 'r')

    counter = 1
    sum_times = 0


    for line in f_id:
        image_name = line.strip('\n')
        image_base_name = image_name.split('.')[0]
        # read image
        img_path = os.path.join(args.dataset_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            break
        img_reverse = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img_out = transform(img_reverse).unsqueeze(0)

        #inference starting...
        start_time = time.time()
        valImg = img_out.to(device)
        out, cx1, cx2 = net(valImg)
        elapsed_time = time.time() - start_time
        sum_times += elapsed_time
        print('Elapsed time: %.04f for image num %03d' % (elapsed_time, counter))
        out = F.log_softmax(out, dim=1)
        pre_label = out.max(dim=1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        cv2.imwrite(args.save_dir + image_base_name + '.png', pre)
        counter += 1

    f_id.close()
    print('Average time per image: %.5f' % (sum_times / num_imgs))


if __name__ == '__main__':
    main()
