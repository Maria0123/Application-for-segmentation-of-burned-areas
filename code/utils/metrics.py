#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
from medpy import metric

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
    return iou
