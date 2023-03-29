import os
import sys
path = os.getcwd()
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
import numpy as np
from sklearn.svm import SVC
import imp
from tqdm import tqdm
    
import __init__ as booger
from modules.SalsaNext_simsiam import *
from modules.simsiam import *

class PointNet(nn.Module):
    def __init__(self, hparams):
        super(PointNet, self).__init__()
        self.hparams = hparams
        self.emb_dims = hparams["emb_dims"]

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()

        return x

def load_model(weight_path, model):
    state_dict = model.state_dict()

    ckpt = torch.load(weight_path, map_location="cpu")
    pretrained_dict = ckpt["state_dict"]

    for key in state_dict:
        if "online_network." + key in pretrained_dict:
            state_dict[key] = pretrained_dict["online_network."+key]
            print(key)

    model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == "__main__":
    # model = SalsaNextEncoder(input_size=10,pred_dim=256)
    # state_dict = model.state_dict()
    # # ckpt = torch.load('checkpoint/pretrained.pth.tar', map_location="cpu")
    # ckpt = torch.load('checkpoint/align_x2.pth.tar', map_location="cpu")
    # pretrained_dict = ckpt["state_dict"]

    # for key in state_dict:
    #     if 'encoder.0.'+key in pretrained_dict:
    #         state_dict[key] = pretrained_dict['encoder.0.'+key]

    # model.load_state_dict(state_dict, strict=True)
    
    model = SalsaSeg('checkpoint/align_x2.pth.tar',10,256)
    
    # hparam = {"model.use_xyz": True, "emb_dims": 1024, "dropout":True, "num_points": 2048, "k": 40}
    # model = PointNet(hparam)
    # input = torch.rand(10,3,50)
    # print(model(input).shape)

    