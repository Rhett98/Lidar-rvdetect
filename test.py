import os
import sys
path = os.getcwd()
sys.path.append(path)

import torch
import argparse
import yaml
import numpy as np
from sklearn.svm import SVC
import imp
from tqdm import tqdm
    
import __init__ as booger
from modules.SalsaNext_simsiam import *
from modules.simsiam import *


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
    model = SalsaNextEncoder()
    state_dict = model.state_dict()
    # ckpt = torch.load('checkpoint/pretrained.pth.tar', map_location="cpu")
    ckpt = torch.load('checkpoint/align_x2.pth.tar', map_location="cpu")
    pretrained_dict = ckpt["state_dict"]

    for key in state_dict:
        if 'encoder.0.'+key in pretrained_dict:
            state_dict[key] = pretrained_dict['encoder.0.'+key]

    model.load_state_dict(state_dict, strict=True)

    