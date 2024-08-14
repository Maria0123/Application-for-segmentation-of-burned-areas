import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
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
parser.add_argument('--with_stats', type=bool,  default=False, help='net stats')

args = parser.parse_args()

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

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False, with_stats=False):
        model = net_factory(net_type=args.model, in_chns=12,
                            class_num=num_classes, with_stats = with_stats)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model_supervised = create_model(with_stats = args.with_stats)
    model_unsupervised = create_model(with_stats = args.with_stats)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = CaBuAr(base_dir=args.root_path, split="train", num=None)
    db_val = CaBuAr(base_dir=args.root_path, split="val")

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

    model_supervised.train()
    model_unsupervised.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer1 = optim.AdamW(model_supervised.parameters(), lr=base_lr, weight_decay=0.01)
    optimizer2 = optim.AdamW(model_unsupervised.parameters(), lr=base_lr, weight_decay=0.01)
    
    ce_loss = CrossEntropyLoss()
    mse_loss = MSELoss()
    dc_loss = losses.DiceLoss(num_classes)

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
            
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            labeled_volume_batch = volume_batch[:args.labeled_bs]
            labeled_label_batch = label_batch[:args.labeled_bs]

            outputs1  = model_supervised(labeled_volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)              
            
            outputs2  = model_supervised(labeled_volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)            
            
            outputs3 = model_unsupervised(unlabeled_volume_batch)
            pseudo_labels = torch.softmax(outputs3, dim=1)

            unsupervised_labels = model_supervised(unlabeled_volume_batch)
            unsupervised_pseudo_labels = torch.softmax(unsupervised_labels, dim=1)

            # loss
            consistency_weight = get_current_consistency_weight(iter_num)
            drop_consistency_weight = 1.0 # get_current_consistency_weight(iter_num)

            loss_mse = mse_loss(outputs1, outputs2)
            drop_loss = drop_consistency_weight * loss_mse

            loss_ce_1 = ce_loss(outputs1, labeled_label_batch.squeeze())
            loss_dc_1 = dc_loss(outputs_soft1, labeled_label_batch)
            model_supervised_loss = 0.5 * (loss_ce_1 + loss_dc_1)

            loss_ce_2 = ce_loss(outputs2, labeled_label_batch.squeeze())
            loss_dc_2 = dc_loss(outputs_soft2, labeled_label_batch)
            model_supervised_2_loss = 0.5 * (loss_ce_2 + loss_dc_2)

            pseudo_labels_loss = losses.compute_kl_loss(unsupervised_pseudo_labels, pseudo_labels)
            model_unsupervised_loss = consistency_weight * pseudo_labels_loss

            loss = model_supervised_loss + model_supervised_2_loss + drop_loss + model_unsupervised_loss

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
            writer.add_scalar(
                'consistency_weight/drop_consistency_weight', drop_consistency_weight, iter_num)
            
            if args.with_stats:
                writeNetStats(model_supervised, model_unsupervised, writer, iter_num)
            
            writer.add_scalar('loss/loss_mse',
                              loss_mse, iter_num)
            writer.add_scalar('loss/drop_loss',
                              drop_loss, iter_num)
            
            writer.add_scalar('loss/loss_ce_1',
                              loss_ce_1, iter_num)            
            writer.add_scalar('loss/loss_dc_1',
                              loss_dc_1, iter_num)   
            writer.add_scalar('loss/model_supervised_loss',
                              model_supervised_loss, iter_num)   
                      
            writer.add_scalar('loss/loss_ce_2',
                              loss_ce_2, iter_num)
            writer.add_scalar('loss/loss_dc_2',
                              loss_dc_2, iter_num)
            writer.add_scalar('loss/model_supervised_2_loss',
                              model_supervised_2_loss, iter_num)            
            
            writer.add_scalar('loss/pseudo_labels_loss',
                              pseudo_labels_loss, iter_num)   
            writer.add_scalar('loss/model_unsupervised_loss',
                              model_unsupervised_loss, iter_num) 
                                    
            writer.add_scalar('loss/loss',
                              loss.item(), iter_num) 
            
            logging.info('iter %d : supervised %f supervised_2 %f drop %f unsupervised %f' % (iter_num, model_supervised_loss, model_supervised_2_loss, drop_loss, model_unsupervised_loss))

            if iter_num % 100 == 0:
                image = volume_batch[0, 2:4, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model_supervised_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model_unsupervised_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model_supervised.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_cbr(
                        sampled_batch["image"], sampled_batch["label"], model_supervised, classes=num_classes)
                    metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)
                performance1 = np.mean(metric_list, axis=0)

                writer.add_scalar('metric_val/supervised_dice', performance1[0], iter_num)
                writer.add_scalar('metric_val/supervised_precision', performance1[1], iter_num)
                writer.add_scalar('metric_val/supervised_recall', performance1[2], iter_num)
                writer.add_scalar('metric_val/supervised_f1', performance1[3], iter_num)
                writer.add_scalar('metric_val/supervised_accuracy', performance1[4], iter_num)
                writer.add_scalar('metric_val/supervised_iou', performance1[5], iter_num)

                logging.info(
                    'model_supervised iteration %d : dice: %f precision: %f recall: %f f1: %f accuracy: %f iou: %f' % 
                        (iter_num, performance1[0], performance1[1], performance1[2], performance1[3], performance1[4], performance1[5]))
  
                performance1_mean = performance1[3] # np.mean(performance1)

                if performance1_mean > best_performance1:
                    best_performance1 = performance1_mean
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_supervised_{}_{}_{}.pth'.format(
                                                      iter_num, round(performance1[0], 4), round(performance1_mean, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model_supervised.pth'.format(args.model))
                    torch.save(model_supervised.state_dict(), save_mode_path)
                    torch.save(model_supervised.state_dict(), save_best)

                model_supervised.train()

                model_unsupervised.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_cbr(
                        sampled_batch["image"], sampled_batch["label"], model_unsupervised, classes=num_classes)
                    metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)
                performance2 = np.mean(metric_list, axis=0) 

                writer.add_scalar('info_val/unsupervised_dice', performance2[0], iter_num)
                writer.add_scalar('info_val/unsupervised_precision', performance2[1], iter_num)
                writer.add_scalar('info_val/unsupervised_recall', performance2[2], iter_num)
                writer.add_scalar('info_val/unsupervised_f1', performance2[3], iter_num)
                writer.add_scalar('info_val/unsupervised_accuracy', performance2[4], iter_num)
                writer.add_scalar('info_val/unsupervised_iou', performance2[5], iter_num)

                logging.info(
                    'model_unsupervised iteration %d : dice: %f precision: %f recall: %f f1: %f accuracy: %f iou: %f' % 
                        (iter_num, performance2[0], performance2[1], performance2[2], performance2[3], performance2[4], performance2[5]))
                 
                performance2_mean = performance1[3] # np.mean(performance2)

                if performance2_mean > best_performance2:
                    best_performance2 = performance2_mean
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_unsupervised_{}_{}_{}.pth'.format(
                                                      iter_num, round(performance2[0], 4), round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model_unsupervised.pth'.format(args.model))
                    torch.save(model_unsupervised.state_dict(), save_mode_path)
                    torch.save(model_unsupervised.state_dict(), save_best)

                model_unsupervised.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model_supervised_' + str(iter_num) + '.pth')
                torch.save(model_supervised.state_dict(), save_mode_path)
                logging.info("save model_supervised to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model_unsupervised_' + str(iter_num) + '.pth')
                torch.save(model_unsupervised.state_dict(), save_mode_path)
                logging.info("save model_unsupervised to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
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