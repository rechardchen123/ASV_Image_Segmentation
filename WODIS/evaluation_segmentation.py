#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:49
# @File  : evaluation_segmentation.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
from __future__ import division
import numpy as np

'''
Evaluation functions: confusion matrix, IoU calculation. For the image segmentation, the IoU is the core evaluation 
function to identify the effectiveness of algorithms. 
The function of 'calc_semantic_segmentation_iou' is to implement the IoU algorithm.
'''


def cal_semantic_segmentation_confusion(pre_labels, gt_labels):
    '''
    Collect a confusion matrix
    :param pre_labels: predicted labels
    :param gt_labels: ground truth labels
    :return: confusion matrix
    '''
    pre_labels = iter(pre_labels)
    gt_labels = iter(gt_labels)

    n_class = 3  # the segmentation categoires:[sea, sky, objects]
    confusion = np.zeros((n_class, n_class), dtype=np.int64)  # (3, 3)
    for pre_label, gt_label in zip(pre_labels, gt_labels):
        if pre_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two...')
        if pre_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same dimension.')

        pre_label = pre_label.flatten()
        gt_label = gt_label.flatten()

        # dynamically expand the confusion matrix if necessary
        lb_max = np.max((pre_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # count statistics from valud pixels
        mask = gt_label >= 0
        confusion += np.bincount(n_class * gt_label[mask].astype(int) + pre_label[mask],
                                 minlength=n_class ** 2).reshape((n_class, n_class))

    for iter_ in (pre_labels, gt_labels):
        # this code assumes any iterator doses not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same...')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    '''Calculate Intersection over Union with a given confusion matrix.'''
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou[:-1]


def eval_semantic_segmentation(pred_labels, gt_labels):
    '''Evaluate metrics used in semantic segmentation'''
    confusion = cal_semantic_segmentation_confusion(pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy[:-1])}
