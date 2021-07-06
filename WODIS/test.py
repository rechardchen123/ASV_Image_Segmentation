#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:48
# @File  : test.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from evaluation_segmentation import eval_semantic_segmentation
from dataset import ASV_ImageDataSet
from WODIS import WODIS_model
import cfg

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

BATCH_SIZE = 4
miou_list = [0]

ASV_test = ASV_ImageDataSet([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(ASV_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

net = WODIS_model(False,3)
net.eval()
net.to(device)
net.load_state_dict(t.load('WODIS_weights.pth'))  # the trained weights

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0

for i, sample in enumerate(test_data):
    data = Variable(sample['img'].to(device))
    label = Variable(sample['label'].to(device))
    out, cx1, cx2 = net(data)
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = label.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_matrix = eval_semantic_segmentation(pre_label, true_label)
    train_acc = eval_matrix['mean_class_accuracy'] + train_acc
    train_miou = eval_matrix['miou'] + train_miou
    train_mpa = eval_matrix['pixel_accuracy'] + train_mpa
    if len(eval_matrix['class_accuracy']) < 12:
        eval_matrix['class_accuracy'] = 0
        train_class_acc = train_class_acc + eval_matrix['class_accuracy']
        error += 1
    else:
        train_class_acc = train_class_acc + eval_matrix['class_accuracy']

    print(eval_matrix['class_accuracy'], '==================', i)

epoch_str = ('test_acc:{:.5f}, test_miou:{:.5f}, test_mpa:{:.5f}, test_class_acc:{:.5f}'.format(
    train_acc / (len(test_data) - error),
    train_miou / (len(test_data) - error),
    train_mpa / (len(test_data) - error),
    train_class_acc / (len(test_data) - error)
))

if train_miou / (len(test_data) - error) > max(miou_list):
    miou_list.append(train_miou / (len(test_data) - error))
    print(epoch_str + '===============last')
