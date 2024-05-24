import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import utils
from dataloaders.CaBuAr import CaBuAr, RandomFlip, RandomNoise, ToTensor
from networks.net_factory import net_factory
from utils import losses, ramps
from val_2D import test_single_volume_cbr

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/CaBuAr', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CaBuAr/Unet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=42, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--alpha_ce', type=float,  default=0.5, help='dice loss weight')
parser.add_argument('--alpha_hd', type=float,  default=0.5, help='hd95 loss weight')

args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=12,
                            class_num=num_classes)

        return model

    model = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = CaBuAr(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomNoise(),
        RandomFlip(),
        ToTensor()
    ]))
    db_val = CaBuAr(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    print("Total silices is: {}".format(total_slices))

    trainloader = DataLoader(db_train, batch_size=batch_size,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.DiceLoss(num_classes)
    hd95_loss = losses.HD95Loss(num_classes)
    loss_ce = 0.0
    loss_hd = 0.0
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            if torch.backends.mps.is_available():
                volume_batch, label_batch = volume_batch.to(torch.float32).to("mps"), label_batch.to(torch.float32).to("mps")
            else:
                volume_batch, label_batch = volume_batch.to(torch.float32).cuda(), label_batch.to(torch.float32).cuda()
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            if args.alpha_ce > 0:
                loss_ce = dice_loss(
                    outputs_soft, label_batch)
            if args.alpha_hd > 0:
                loss_hd = hd95_loss(
                    outputs_soft, label_batch)
            loss = args.alpha_ce * loss_ce + args.alpha_hd * loss_hd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss_dice', loss, iter_num)

            logging.info(
                'iteration : %d loss: %f loss_dice: %f loss_hd95: %f' %
                (iter_num, loss.item(), loss_ce, loss_hd))

            if iter_num % 20 == 0:
                image = volume_batch[0, 2:4, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 300 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_cbr(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_f1'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)
                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95[1], iter_num)
                writer.add_scalar('info/val_mean_jc', mean_hd95[2], iter_num)
                writer.add_scalar('info/val_mean_f1', mean_hd95[3], iter_num)
                
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f mean_jc : %f mean_f1 : %f' % 
                        (iter_num, performance, mean_hd95[1], mean_hd95[2], mean_hd95[3]))
                model.train()

            if iter_num % 300 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}".format(
        args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)