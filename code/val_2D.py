import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    dice = metric.binary.dc(pred, gt)
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    
    if pred.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        jc = metric.binary.jc(pred, gt)
        return dice, hd95, jc, f1
    else:
        return dice, 100, 0, f1


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


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

        if torch.backends.mps.is_available():
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().to("mps")
        else:
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_cbr(image, label, net, classes, patch_size=[12, 512, 512]):
    # Info dla mnie: jeśli będziesz miała więcej ni 1 batch na obrót
    # to będzie trzeba cofnąć do tego co  było poprzednio - tzn pętli.
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