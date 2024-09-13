import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from tensorboardX import SummaryWriter

# from networks.efficientunet import UNet
from utils.metrics import intersection_over_union
from networks.net_factory import net_factory

from torchmetrics import JaccardIndex

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/CaBuAr', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CaBuAr/Unet/batch8', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=0,
                    help='labeled data')
parser.add_argument('--scenario', type=str,  default=None, help='scenario B1, B3, B4, or None')

def calculate_metric_percase(pred, gt):
    pred = np.ravel(pred)
    gt = np.ravel(gt)

    precision = precision_score(pred, gt)
    recall = recall_score(pred, gt)
    f1 = f1_score(pred, gt)
    accuracy = accuracy_score(pred, gt)
    iou = intersection_over_union(pred, gt)

    Ji = JaccardIndex(task="multiclass", num_classes=2)
    Ji_result = Ji(torch.tensor(pred), torch.tensor(gt))

    return precision, recall, f1, accuracy, iou, Ji_result

def test_single_volume(case, net, test_save_path, FLAGS, writer, i=0):
    def match_scenatio(image):
        match (FLAGS.scenario):
            case "B1":
                indices = [6]
                image = image[indices, :, :]
            case "B3":
                indices = [1, 10, 11]
                image = image[indices, :, :]
            case "B4":
                indices = [1, 6, 10, 11]
                image = image[indices, :, :]
            case _:
                image = image
        return image

    h5f = h5py.File(FLAGS.root_path + "/data/slices/{}.h5".format(case), 'r')
    image_oryginal = h5f['image'][:]
    image = match_scenatio(image_oryginal)
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    slice = image
    input = torch.from_numpy(slice).unsqueeze(
        0).float()
    if torch.backends.mps.is_available():
        input = input.to(torch.float32).to("mps")
    else:
        input = input.cuda()

    net.eval()
    with torch.no_grad():
        net_out = net(input)
        out = torch.argmax(torch.softmax(
            net_out, dim=1), dim=1)
        out = out.cpu().detach().numpy()
        prediction = out

    metric = calculate_metric_percase(prediction, label)

    compare_image = np.zeros((3,128,128))

    for k in range(128):
        for j in range(128):
            if prediction[0][k][j] == label[0][k][j]:
                compare_image[0][k][j] = prediction[0][k][j]
                compare_image[1][k][j] = prediction[0][k][j]
                compare_image[2][k][j] = prediction[0][k][j]
            elif prediction[0][k][j] > label[0][k][j]:
                compare_image[0][k][j] = 1
                compare_image[1][k][j] = 0
                compare_image[2][k][j] = 0
            else:
                compare_image[0][k][j] = 0
                compare_image[1][k][j] = 1
                compare_image[2][k][j] = 0


    writer.add_image("Image", image_oryginal[1:3, :, :], i)
    writer.add_image("Prediction", prediction * 50, i)
    writer.add_image("GroundTruth", label * 50, i)
    writer.add_image("CompareImage", compare_image, i)

    writer.add_scalar('info/val_mean_precision', metric[0], i)
    writer.add_scalar('info/val_mean_recall', metric[1], i)
    writer.add_scalar('info/val_mean_f1', metric[2], i)
    writer.add_scalar('info/val_mean_accuracy', metric[3], i)
    writer.add_scalar('info/val_mean_iou', metric[4], i)

    return metric
        
def Inference(FLAGS):
    def get_chanels():
        match (FLAGS.scenario):
            case 'B1':
                return 1
            case 'B3':
                return 3
            case 'B4':
                return 4
            case _:
                return 12
        
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/{}".format(
        FLAGS.exp, FLAGS.model)
    test_save_path = "../model/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    writer = SummaryWriter(test_save_path, max_queue=110)

    net = net_factory(net_type=FLAGS.model, in_chns=get_chanels(),
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    for i, case in tqdm(enumerate(image_list)):
        first_metric = test_single_volume(
            case, net, test_save_path, FLAGS, writer, i)
        first_total += np.asarray(first_metric)
    avg_metric = first_total / len(image_list)
    
    writer.close()

    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
