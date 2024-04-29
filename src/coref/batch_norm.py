import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

m = nn.BatchNorm1d(3, affine=False)
# m = nn.BatchNorm2d(3, affine=False)
i = torch.randn(2, 3)
output = m(i)
a = 1
