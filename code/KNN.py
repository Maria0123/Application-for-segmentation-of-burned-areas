import argparse
import logging
import os
import random
import shutil
import sys
from tqdm import tqdm

import h5py

import numpy as np
from sklearn import neighbors
from tensorboardX import SummaryWriter

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from utils.metrics import intersection_over_union


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/CaBuAr', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CaBuAr/Unet', help='experiment_name')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=42, help='random seed')

args = parser.parse_args()

def calculate_metric_percase(pred, gt):
    pred = np.ravel(pred)
    gt = np.ravel(gt)

    precision = precision_score(pred, gt)
    recall = recall_score(pred, gt)
    f1 = f1_score(pred, gt)
    accuracy = accuracy_score(pred, gt)
    iou = intersection_over_union(pred, gt)
    
    return precision, recall, f1, accuracy, iou

def train(args):
    
    with open(args.root_path + "/train_slices.list", "r") as f1:
        train_list_idx = f1.readlines()
    train_list_idx = [item.replace("\n", "") for item in train_list_idx]

    train_list = []
    train_label_list = []

    for idx in train_list_idx:
        h5f = h5py.File(args.root_path + "/data/slices/{}.h5".format(idx), "r")

        image = h5f["image"][:]
        label = h5f["label"][:]
        
        train_list.append(image)
        train_label_list.append(label)
    
    train_list = np.array(train_list)
    train_list = train_list.reshape(208, -1)

    train_label_list = np.array(train_label_list)
    train_label_list = train_label_list.reshape(208, -1)

    print(np.shape(train_list), np.shape(train_label_list))

    print("Training started")
    clf = neighbors.KNeighborsClassifier(n_neighbors=100)
    clf.fit(train_list, train_label_list)
    print("Training completed")

    with open(args.root_path + "/test.list", "r") as f1:
        test_list_idx = f1.readlines()
    test_list_idx = [item.replace("\n", "") for item in test_list_idx]

    test_list = []
    test_label_list = []

    for idx in test_list_idx:
        h5f = h5py.File(args.root_path + "/data/slices/{}.h5".format(idx), "r")

        image = h5f["image"][:]
        label = h5f["label"][:]

        test_list.append(image)
        test_label_list.append(label)

    test_list = np.array(test_list)
    test_list = test_list.reshape(106, -1)

    test_label_list = np.array(test_label_list)
    test_label_list = test_label_list.reshape(106, -1)
    
    predicted_mask = clf.predict(test_list)

    metric = [0,0,0,0,0]
    for X, y in zip(predicted_mask, test_label_list):
        results = calculate_metric_percase(X, y)
        metric = [x + y for x, y in zip(metric, results)]

    metric = [ x / len(test_list) for x in metric]
    print(metric)


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args)