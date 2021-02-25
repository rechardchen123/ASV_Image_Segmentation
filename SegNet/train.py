#!/usr/bin/env python3
# -*- coding: utf-8 -*
# **************************************************
# @Time  : 18/12/2020 14:53
# @File  : bisenet.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# **************************************************
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import ASV_ImageDataSet
from evaluation_segmentation import eval_semantic_segmentation
from segnet1 import SegNet
import cfg

if t.cuda.is_available():
    device = t.device('cuda')
else:
    device = t.device('cpu')

ASV_train = ASV_ImageDataSet([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
ASV_val = ASV_ImageDataSet([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(ASV_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
val_data = DataLoader(ASV_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

segnet = SegNet(3, 3)
segnet = segnet.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(segnet.parameters(), lr=1e-4)


def train(model):
    best = [0]
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        print("Epoch is [{}/{}]".format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0

        for i, sample in enumerate(train_data):
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))

            out, out1 = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metric = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metric['mean_class_accuracy']
            train_miou += eval_metric["miou"]
            train_class_acc += eval_metric["class_accuracy"]

            print("|batch[{}/{}] |batch_loss {:.8f}".format(i + 1, len(train_data), loss.item()))

        metric_description = '|Train Acc |： {:.5f}| Train Mean IoU|: {:.5f}\n |Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data),
        )

        print(metric_description)
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            t.save(net.state_dict(), '{}.pth'.format(epoch))


def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou

    cur_time = datetime.now()
    h, reminder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(reminder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|:{:.5f}\n|Valid Mean Iou|:{:.5f}\n|Valid Class Acc|:{:}'.format(
        eval_loss / len(train_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)
    ))
    print(val_str)
    print(time_str)


if __name__ == '__main__':
    train(segnet)
