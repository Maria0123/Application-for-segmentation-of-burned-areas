import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.CaBuAr import CaBuAr
from dataloaders.dataset import TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, ramps
from val import test_single_volume_cbr

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/CaBuAr', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CaBuAr/MT', help='experiment_name')
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
                    default=10.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument('--scenario', type=str,  default=None, help='scenario B1, B3, B4, or None')
parser.add_argument('--random', type=bool,  default=False, help='test on random data')

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
        
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "CaBuAr" in dataset:
        ref_dict = {"16": 35, "30": 70, "45": 94}
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

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=chanels,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = CaBuAr(base_dir=args.root_path, split="train", num=None, scenario=args.scenario)
    db_val = CaBuAr(base_dir=args.root_path, split="val", scenario=args.scenario)

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    
    if args.random:
        indices = list(range(total_slices))
        labeled_idxs = random.sample(indices, labeled_slice)
        unlabeled_idxs = [i for i in indices if i not in labeled_idxs]
    else:
        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, total_slices))

    print("Total silices is: {}, labeled slices is: {}, unlabeld slices is: {}".format(
        total_slices, len(labeled_idxs), len(unlabeled_idxs))) 
    
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    ce_loss = CrossEntropyLoss()                      
    dc_hd_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance_ce = 0.0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            if torch.backends.mps.is_available():
                volume_batch, label_batch = volume_batch.to(torch.float32).to("mps"), label_batch.to(torch.float32).to("mps")
            else:
                volume_batch, label_batch = volume_batch.cuda(), label_batch.type(torch.LongTensor).cuda()
            
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise            
            
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce   = ce_loss(outputs[:args.labeled_bs], 
                                label_batch[:args.labeled_bs].squeeze())
            loss_dc_hd = dc_hd_loss(outputs_soft[:args.labeled_bs], 
                                  label_batch[:args.labeled_bs])
            
            supervised_loss = 0.5 * (loss_dc_hd + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num)
            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:]-ema_output_soft)**2)
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dc_hd', loss_dc_hd, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('info/supervised_loss',
                              supervised_loss, iter_num)
            writer.add_scalar('info/unsupervised_loss',
                              consistency_weight * consistency_loss, iter_num)
            
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dc_hd: %f' % 
                (iter_num, loss.item(), loss_ce.item(), loss_dc_hd.item()))

            if iter_num % 20 == 0:
                image = sampled_batch['oryginal'][0, 2:4, :, :]
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
                performance = np.mean(metric_list, axis=0)

                writer.add_scalar('info/val_mean_dice', performance[0], iter_num)
                writer.add_scalar('info/val_mean_hd95', performance[1], iter_num)
                writer.add_scalar('info/val_mean_jc', performance[2], iter_num)
                writer.add_scalar('info/val_mean_f1', performance[3], iter_num)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f mean_jc : %f mean_f1 : %f' % 
                        (iter_num, performance[0], performance[1], performance[2], performance[3]))
                
                performance_ce = performance[0]
                if performance_ce > best_performance_ce:
                    best_performance_ce = performance_ce
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_ce_{}.pth'.format(
                                                      iter_num, round(best_performance_ce, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

        scheduler.step()

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