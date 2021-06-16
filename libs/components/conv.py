import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNNLayer(nn.Module):
    '''
    TDNN + BN + activation
    '''
    def __init__(self, input_channel, output_channel, context, padding = 0, stride = 1, bn_first = True):
        super(TDNNLayer, self).__init__()
        kernel_size = len(context)
        if len(context) > 1:
            dilation = (context[-1] - context[0]) // (len(context) - 1)
        else:
            dilation = 1
        self.context_layer = nn.Conv1d(
                input_channel,
                output_channel,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation
                )
        self.bn = nn.BatchNorm1d(output_channel)
        #  self.activation = nn.LeakyReLU(negative_slope = 0.2)
        self.activation = nn.ReLU()
        #  self.bn_first = bn_first

    def forward(self, x):
        x = self.context_layer(x)
        #  if self.bn_first:
            #  x = self.bn(x)
            #  x = self.activation(x)
        #  else:
        x = self.activation(x)
        x = self.bn(x)
        return x

class Conv3x3(nn.Conv2d):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
        super(Conv3x3, self).__init__(in_planes, out_planes, kernel_size = 3,
                                      stride = stride, padding = dilation, groups = groups,
                                      bias = False, dilation = dilation)

class Conv1x1(nn.Conv2d):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super(Conv1x1, self).__init__(in_planes, out_planes, kernel_size = 1,
                                      stride = stride, bias = False)

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
