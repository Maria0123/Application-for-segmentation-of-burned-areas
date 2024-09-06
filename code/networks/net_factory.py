import torch
from networks.hrnet import get_pose_net
from networks.segnet import SegNet
from networks.unet import UNet


def net_factory(net_type="unet", in_chns=1, class_num=3, with_stats=False):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num, with_stats=with_stats)
    elif net_type == "segnet":
        net = SegNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "hrnet":
        net = get_pose_net()
    else:
        net = None

    if net != None:
        if torch.backends.mps.is_available():
            net = net.to("mps")
        else:
            net = net.cuda()
    return net
