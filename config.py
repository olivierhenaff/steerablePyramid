import torch

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32
