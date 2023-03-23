import torch
import torch.nn as nn

input1 = torch.randn(1,256)
input2 = torch.randn(1,256)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(output)