import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from tensorboardX import SummaryWriter

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/CaBuArRaw', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CaBuAr/Unet/batch8', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=0,
                    help='labeled data')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    
    if pred.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)

        precision = metric.binary.precision(pred, gt)
        recall = metric.binary.recall(pred, gt)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)  
        return dice, hd95, jc, f1

    return dice, 100, jc, 0

def test_single_volume(case, net, test_save_path, FLAGS, writer, i=0):
    h5f = h5py.File(FLAGS.root_path + "/data/slices/{}.h5".format(case), 'r')
    image = h5f['image'][:]
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

    writer.add_image("Image", image[1:3, :, :], i)
    writer.add_image("Prediction", prediction * 50, i)
    writer.add_image("GroundTruth", label * 50, i)

    writer.add_scalar('info/val_mean_dice', metric[0], i)
    writer.add_scalar('info/val_mean_hd95', metric[1], i)
    writer.add_scalar('info/val_mean_jc', metric[2], i)
    writer.add_scalar('info/val_mean_f1', metric[3], i)

    return metric


def Inference(FLAGS):
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
    writer = SummaryWriter(test_save_path)

    net = net_factory(net_type=FLAGS.model, in_chns=12,
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
