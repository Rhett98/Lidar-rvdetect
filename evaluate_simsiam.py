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

    model.load_state_dict(state_dict, strict=False)
    return model


def fetch_represent(loader, model):
    representations = None
    labels = None
    batch_num = len(loader)
    for batch_id, data in tqdm(enumerate(loader)):
        if batch_id == 500:
            break
        if (batch_id + 1) % 100 == 0:
            print("%d/%d" % (batch_id+1, batch_num))
        pc, label = data
        pc,_ = torch.split(pc, 10, dim=1)
        # print(pc.shape, label.shape)
        pc = pc.to(device)
        # pc = pc.permute(0, 2, 1)
        with torch.no_grad():
            representation = model(pc).cpu().numpy()
            # print(representation.shape)
        if representations is None:
            representations = representation
            labels = label
        else:
            representations = np.concatenate([representations, representation], 0)
            labels = np.concatenate([labels, label], 0)
    return representations, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", default='checkpoint/pretrained.pth.tar', type=str, help="the ckpt path to load")
    parser.add_argument("--xyz_only", default=1, type=str, help="whether to only use xyz-coordinate for evaluation")
    parser.add_argument("--num_points", default=2048, type=int)
    parser.add_argument("--k", default=40, type=int, help="choose gpu")
    parser.add_argument("--dropout", default=0.5, type=float, help="choose gpu")
    parser.add_argument("--emb_dims", default=1024, type=int, help="dimension of hidden embedding")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size of dataloader")
    parser.add_argument("--gpu_num", default=0, type=int, help="choose gpu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hparam = load_hparam(args.model_yaml)
    hparam = {"model.use_xyz": True, "emb_dims": args.emb_dims, "dropout":args.dropout, "num_points": args.num_points, "k": args.k}

    # load model
    model = SalsaNextEncoder()
    state_dict = model.state_dict()
    ckpt = torch.load(args.weight, map_location="cpu")
    pretrained_dict = ckpt["state_dict"]
    for key in state_dict:
        if 'encoder.0.'+key in pretrained_dict:
            state_dict[key] = pretrained_dict['encoder.0.'+key]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    # load dataset
    ARCH = yaml.safe_load(open('config/simsiam.yml', 'r'))
    DATA = yaml.safe_load(open('config/labels/local-test.yaml', 'r'))
    data = '../dataset'
    parserModule = imp.load_source("parserModule",
                    f"{booger.TRAIN_PATH}/common/dataset/{DATA['name']}/parser_simsiam.py")

    train_dataset = parserModule.KittiRV('train', ARCH, DATA, data,
                                gt=True,transform=False,drop_few_static_frames=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=True)

    valid_dataset = parserModule.KittiRV('valid', ARCH, DATA, data,
                                    gt=True,transform=False,drop_few_static_frames=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=True)

    print("Fetch Train Data Representation")
    train_represent, train_label = fetch_represent(train_loader, model)

    print("Fetch Val Data Representation")
    val_represent, val_label = fetch_represent(valid_loader, model)

    svc = SVC(kernel="linear", verbose=True)
    svc.fit(train_represent, train_label)

    score = svc.score(val_represent, val_label)
    print("Val Accuracy:", score)
