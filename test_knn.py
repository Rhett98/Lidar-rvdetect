import imp
import yaml
from tqdm import tqdm
    
import torch
import torch.nn as nn
import __init__ as booger
from modules.KNN import *
from modules.SalsaNext_simsiam import *
from modules.simsiam import *

# load model
model = SimSiam(SalsaNextEncoder(), 1024, 256)
model = model.cuda()
backbone = nn.Sequential(model.align,
                         model.backbone)

# load dataset
ARCH = yaml.safe_load(open('config/simsiam.yml', 'r'))
DATA = yaml.safe_load(open('config/labels/local-test.yaml', 'r'))
data = '../dataset'
parserModule = imp.load_source("parserModule",
                f"{booger.TRAIN_PATH}/common/dataset/{DATA['name']}/parser_simsiam.py")
train_dataset = parserModule.KittiRV('train', ARCH, DATA, data,
                                gt=True,transform=False,drop_few_static_frames=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=True)

test_dataset = parserModule.KittiRV('test', ARCH, DATA, data,
                                gt=True,transform=False,drop_few_static_frames=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=True)

# KNN evaluation
param = ARCH['post']['KNN']['params']
n_classes = 2
accuracy = KNN(param, n_classes)        
epoch_dict = {"accuracy":accuracy}
print(epoch_dict)