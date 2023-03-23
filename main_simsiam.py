#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import random
import shutil
import time
import warnings
import imp
import yaml
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast, grad_scaler
from tensorboardX import SummaryWriter as Logger

import __init__ as booger
from modules.SalsaNext_simsiam import *
from modules.simsiam import *

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch_cfg', '-ac',
                    type=str,
                    required=True,
                    help='Architecture yaml cfg file. See'+ 
                    '/config for sample. No default!',)
parser.add_argument('--data_cfg', '-dc',
                    type=str,
                    required=False,
                    default='config/labels/semantic-kitti-mos.yaml',
                    help='Classification yaml cfg file. See /config/'+
                    'labels for sample.',)

def main():
    args, unparsed = parser.parse_known_args()
    seed = None
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting ')
        # Simply call main_worker function
    main_worker(args)

def main_worker(args):
    # open arch config file
    try:
        print("Opening arch config file %s" % args.arch_cfg)
        ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
    # open data config file
    try:
        print("Opening data config file %s" % args.data_cfg)
        DATA = yaml.safe_load(open(args.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
     
    workers = ARCH["train"]["workers"]
    epochs = ARCH["train"]["max_epochs"]
    start_epoch = 0
    resume = 'checkpoint/pretrained.pth.tar' # load chechpoint path
    # resume = None # load chechpoint path
    gpu = 0
    batch_size = ARCH["train"]["batch_size"]
    lr = ARCH["train"]["lr"]
    momentum = ARCH["train"]["momentum"]
    w_decay = ARCH["train"]["w_decay"]
    pred_dim = ARCH["train"]["pred_dim"]
    dim = ARCH["train"]["dim"]
    fix_pred_lr = ARCH["train"]["fix_pred_lr"]
    log = 'log'
    tb_logger = Logger(log + "/tb")

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    # create model
    print("=> creating model")
    model = SimSiam(SalsaNextEncoder(), dim, pred_dim)

    # infer learning rate before changing batch size
    init_lr = lr * batch_size / 256

    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model.half()
    # print(model) # print model 
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total of Trainable Parameters: {}".format(pytorch_total_params))

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(0).half()

    if fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=momentum,
                                weight_decay=w_decay)

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'],strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True

    parserModule = imp.load_source("parserModule",
                                       f"{booger.TRAIN_PATH}/common/dataset/{DATA['name']}/parser_simsiam.py")
    train_dataset = parserModule.KittiRV('train', ARCH, DATA, args.data,
                                        gt=False,transform=False,drop_few_static_frames=False)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, drop_last=True, pin_memory=True, prefetch_factor=True)

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, epochs)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, epochs, tb_logger)

        if epoch % 3 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'salsanext',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, max_epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image_0, image_1 = torch.split(images, 10, dim=1)

        image_0 = image_0.cuda(0, non_blocking=True).half()
        image_1 = image_1.cuda(0, non_blocking=True).half()

        with autocast():
            # compute output and loss
            p1, p2, z1, z2 = model(x1=image_0, x2=image_1)
            a = criterion(p1, z2).mean()
            b = criterion(p2, z1).mean()
            # print('l1:', a, 'l2:',b)
            loss = -(2*a/3 + b/3)
            # print('loss', loss)
            # print('**************')
            # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            losses.update(loss.item(), image_0.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
            print('train time left: ',calculate_estimate(max_epoch,epoch,i,len(train_loader),data_time.avg, batch_time.avg))
    # tensorboard logger
    logger.add_scalar('loss', loss, epoch)

def calculate_estimate(max_epoch, epoch, iter, len_data, data_time_t, batch_time_t):
        estimate = int((data_time_t + batch_time_t) * (len_data * max_epoch - (iter + 1 + epoch * len_data)))
        return str(datetime.timedelta(seconds=estimate))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, max_epoch):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
