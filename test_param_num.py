import torch
from torchvision.models import resnet18, resnet50, resnet101
from modules.SalsaNext_simsiam import *

# model = resnet101()
model = SalsaNextEncoder()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total of Trainable Parameters: {}".format(pytorch_total_params,2))