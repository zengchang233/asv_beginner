import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNNLayer(nn.Module):
    def __init__(self):
        super(TDNNLayer, self).__init__()

    def forward(self):
        pass

class Conv3x3(nn.Module):
    def __init__(self):
        super(Conv3x3, self).__init__()

    def forward(self):
        pass

class Conv1x1(nn.Module):
    def __init__(self):
        super(Conv1x1, self).__init__()

    def forward(self):
        pass

class FTDNNLayer(nn.Module):
    def __init__(self):
        super(FTDNNLayer, self).__init__()

    def forward(self):
        pass



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
