#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import imp
import yaml
import time
from PIL import Image
import collections
import copy
import cv2
import os
import numpy as np

from modules.SalsaNext_simsiam import *
from modules.SalsaNextUncertainty import *
from modules.KNN import KNN


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split,uncertainty,mc=30):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.uncertainty = uncertainty
    self.split = split
    self.mc = mc

    # get the data
    parserModule = imp.load_source("parserModule",
                                       f"{booger.TRAIN_PATH}/common/dataset/{self.DATA['name']}/parser_simsiam.py")
    self.parser = parserModule.KittiRV('train', ARCH, DATA, datadir, True , False, True)

    self.train_loader = torch.utils.data.DataLoader(self.parser, batch_size=ARCH["train"]["batch_size"],
                                                      shuffle=True,num_workers=ARCH["train"]["workers"], drop_last=True, 
                                                      pin_memory=True, prefetch_factor=True)
    self.parser_valid = parserModule.KittiRV('valid', ARCH, DATA, datadir, True)                                                
    self.valid_loader = torch.utils.data.DataLoader(self.parser_valid, batch_size = ARCH["train"]["batch_size"],
                                                      shuffle=True,num_workers=ARCH["train"]["workers"], drop_last=True, 
                                                      pin_memory=True, prefetch_factor=True)

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        self.model = SalsaNext(10,3)
        # self.model = nn.DataParallel(self.model)
        w_dict = torch.load(modeldir + "/SalsaNext_valid_best",
                            map_location=lambda storage, loc: storage)
        self.model.load_state_dict(w_dict['state_dict'], strict=True)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_subset(loader=self.train_loader,
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.valid_loader,
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # # do test set
        # self.infer_subset(loader=self.parser.get_test_set(),
        #                   to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.valid_loader,
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.train_loader,
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    # else:
    #     self.infer_subset(loader=self.parser.get_test_set(),
    #                     to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode
    if not self.uncertainty:
      self.model.eval()
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in,_ = torch.split(proj_in, 10, dim=1)
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()
      
        # compute output
        proj_output = self.model(proj_in)
        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)
        
        if self.post:
            # knn postproc
            print(proj_range.shape)
            unproj_argmax = self.post(proj_range,
                                      unproj_range,
                                      proj_argmax,
                                      p_x,
                                      p_y)
        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
              "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
