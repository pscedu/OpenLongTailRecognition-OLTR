import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

import pdb

class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dim=None, num_classes=None, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())

def create_model(in_dim=None, num_classes=None):
    print('Loading Cosine Norm Classifier.')
    assert in_dim is not None
    assert num_classes is not None
    return CosNorm_Classifier(in_dim=in_dim, num_classes=num_classes)