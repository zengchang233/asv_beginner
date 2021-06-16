import sys
sys.path.insert(0, '../../')

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from libs.components.conv import conv1x1
from libs.components.blocks import ResidualBasicBlock

class ResNet18(nn.Module):
    def __init__(self, opts):
        super(ResNet18, self).__init__()
        block = ResidualBasicBlock # nn.Module type
        norm_layer = None # nn.Module type
        replace_stride_with_dilation = None # List, whose element is bool value
        layers = opts['residual_block_layers']
        self.embedding_dim = opts['embedding_dim']
        self.zero_init_residual = opts['zero_init_residual']
        groups = opts['groups']
        width_per_group = opts['width_per_group']
        pooling = opts['pooling']
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], block)
        self.layer2 = self._make_layer(128, layers[1], block, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], block, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, layers[3], block, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.embedding_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, planes, blocks, block = ResidualBasicBlock,
                    stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def extract_embedding(self, x):
        assert len(x.size()) == 3, 'The shape of input should be BxCxT'
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pooling_x = self.avgpool(x)
        x = torch.flatten(pooling_x, 1)
        x = self.fc(x)

        return x, pooling_x

    def forward(self, x):
        x, _ = self.extract_embedding(x)
        return x

if __name__ == '__main__':
    import yaml
    from yaml import CLoader
    from torchsummary import summary
    f = open('../../conf/model/resnet18.yaml', 'r')
    opts = yaml.load(f, Loader = CLoader)
    f.close()
    net = ResNet18(opts)
    summary(net.cuda(), (161, 300))
