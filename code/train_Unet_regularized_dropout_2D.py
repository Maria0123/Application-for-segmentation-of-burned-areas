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
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.CaBuAr import CaBuAr
from dataloaders.dataset import TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.stats_writer import writeNetStats
from val_2D import test_single_volume_cbr

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/CaBuArRaw', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CaBuArRaw/Regularized_Dropout', help='experiment_name')
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
parser.add_argument('--patch_size', type=list,  default=[128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=42, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=4.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# net stats
parser.add_argument('--with_stats', type=bool,  default=True, help='net stats')

parser.add_argument('--scenario', type=str,  default='B4', help='scenario B1, B3, B4, or None')

args = parser.parse_args()

def get_chanels():
    match (args.scenario):
        case 'B1':
            return 1
        case 'B3':
            return 3
        case 'B4':
            return 4
        case _:
            return 12
        
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "CaBuAr" in dataset:
        ref_dict = {"2": 20, "4": 40, "6": 60, "7": 70, "8": 80, "10": 100}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    chanels = get_chanels()

    def create_model(ema=False, with_stats=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=chanels,
                            class_num=num_classes, with_stats = with_stats)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model(with_stats = args.with_stats)
    model2 = create_model(with_stats = args.with_stats)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = CaBuAr(base_dir=args.root_path, split="train", num=None, scenario=args.scenario)
    db_val = CaBuAr(base_dir=args.root_path, split="val", scenario=args.scenario)

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    
    # labeled_idxs = list(range(0, labeled_slice))
    # unlabeled_idxs = list(range(labeled_slice, total_slices))

    indices = list(range(total_slices))
    labeled_idxs = random.sample(indices, labeled_slice)
    unlabeled_idxs = [i for i in indices if i not in labeled_idxs]

    print("Total silices is: {}, labeled slices is: {}, unlabeld slices is: {}".format(
        total_slices, len(labeled_idxs), len(unlabeled_idxs))) 
    
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer1 = optim.AdamW(model1.parameters(), lr=base_lr, weight_decay=0.01)
    optimizer2 = optim.AdamW(model2.parameters(), lr=base_lr, weight_decay=0.01)
    
    ce_loss = CrossEntropyLoss()
    dc_hd_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            if torch.backends.mps.is_available():
                volume_batch, label_batch = volume_batch.to(torch.float32).to("mps"), label_batch.to(torch.float32).to("mps")
            else:
                volume_batch, label_batch = volume_batch.cuda(), label_batch.type(torch.LongTensor).cuda()
            

            outputs1  = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss_ce_1 = ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].squeeze())
            loss_dc_hd_1 = dc_hd_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs])
            model1_loss = 0.5 * (loss_ce_1 + loss_dc_hd_1)

            loss_ce_2 = ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].squeeze())
            loss_dc_hd_2 = dc_hd_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs])
            model2_loss = 0.5 * (loss_ce_2 + loss_dc_hd_2)

            r_drop_loss = losses.compute_kl_loss(outputs1[args.labeled_bs:], outputs2[args.labeled_bs:])

            loss = model1_loss + model2_loss + consistency_weight * r_drop_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            
            if args.with_stats:
                writeNetStats(model1, model2, writer, iter_num)
            
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/r_drop_loss',
                              r_drop_loss, iter_num)
            writer.add_scalar('loss/loss_ce_model1',
                              loss_ce_1, iter_num)            
            writer.add_scalar('loss/loss_dc_hd_model1',
                              loss_dc_hd_1, iter_num)   
            writer.add_scalar('loss/loss_ce_model2',
                              loss_ce_2, iter_num)            
            writer.add_scalar('loss/loss_dc_hd_model2',
                              loss_dc_hd_2, iter_num) 
            writer.add_scalar('loss/total_loss',
                              loss, iter_num) 
                                    
            logging.info('iteration %d : model1 loss : %f model2 loss : %f r_drop_loss: %f' % (iter_num, model1_loss.item(), model2_loss.item(), r_drop_loss.item()))

            if iter_num % 50 == 0:
                image = sampled_batch['oryginal'][0, 2:4, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 300 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_cbr(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)
                performance1 = np.mean(metric_list, axis=0)

                writer.add_scalar('info/model1_val_mean_dice', performance1[0], iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', performance1[1], iter_num)
                writer.add_scalar('info/model1_val_mean_jc', performance1[2], iter_num)
                writer.add_scalar('info/model1_val_mean_f1', performance1[3], iter_num)

                logging.info(
                    'MODEL1 iteration %d : mean_dice : %f mean_hd95 : %f mean_jc : %f mean_f1 : %f' % 
                        (iter_num, performance1[0], performance1[1], performance1[2], performance1[3]))
  
                performance1 = performance1[0]
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_cbr(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)
                performance2 = np.mean(metric_list, axis=0)

                writer.add_scalar('info/model2_val_mean_dice', performance2[0], iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', performance2[1], iter_num)
                writer.add_scalar('info/model2_val_mean_jc', performance2[2], iter_num)
                writer.add_scalar('info/model2_val_mean_f1', performance2[3], iter_num)

                logging.info(
                    'MODEL2 iteration %d : mean_dice : %f mean_hd95 : %f mean_jc : %f mean_f1 : %f' % 
                        (iter_num, performance2[0], performance2[1], performance2[2], performance2[3]))
  
                performance2 = performance2[0]

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


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