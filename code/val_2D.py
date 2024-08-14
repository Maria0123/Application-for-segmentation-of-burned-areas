import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils.metrics import intersection_over_union


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pred = np.ravel(pred)
    gt = np.ravel(gt)

    precision = precision_score(pred, gt)
    recall = recall_score(pred, gt, zero_division=0.0)
    f1 = f1_score(pred, gt)
    accuracy = accuracy_score(pred, gt)
    iou = intersection_over_union(pred, gt)

    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice, precision, recall, f1, accuracy, iou
    
    return 0, precision, recall, f1, accuracy, iou


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float()
        
        if torch.backends.mps.is_available():
            input = input.to("mps")
        else:
            input = input.cuda()

        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cbr(image, label, net, classes, patch_size=[12, 512, 512]):
    image, label = image.squeeze(0).cpu().detach(
        ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    slice = image
    x, y, z = slice.shape[0], slice.shape[1], slice.shape[2]
    input = torch.from_numpy(slice).unsqueeze(
        0).float()
    if torch.backends.mps.is_available():
        input = input.to("mps")
    else:
        input = input.cuda()

    net.eval()
    with torch.no_grad():
        net_out = net(input)
        out = torch.argmax(torch.softmax(
            net_out, dim=1), dim=1)
        out = out.cpu().detach().numpy()
        prediction = out
    metric_list = []
    metric_list.append(calculate_metric_percase(prediction, label))
    
    return metric_list