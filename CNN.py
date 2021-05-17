import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

inputs = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 5, 5)
pool = nn.MaxPool2d(2)
out = conv1(inputs)
out2 = pool(out)
out.size()
out2.size