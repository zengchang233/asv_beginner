from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import math

class ReLU20(nn.Hardtanh): # relu
    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

def conv3x3(in_planes, out_planes, stride=1): # 3x3卷积，输入通道，输出通道，stride
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module): # 定义block
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, downsample=None): # 输入通道，输出通道，stride，下采样
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = ReLU20(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out # block输出


class SpeakerEmbNet(nn.Module): # 定义resnet
    def __init__(self, opts): # block类型，embedding大小，分类数，maigin大小
        super(SpeakerEmbNet, self).__init__()
        input_dim = opts['input_dim']
        hidden_dim = opts['hidden_dim']
        residual_block_layers = opts['residual_block_layers']
        fc_layers = opts['fc_layers']
        block = BasicBlock
        embedding_size = opts['embedding_dim']
        pooling = opts['pooling']
        self.relu = ReLU20(inplace=True)

        block_layers = []
        for dim, block_layer in zip(hidden_dim, residual_block_layers):
            block_layers.append(nn.Conv2d(input_dim, dim, kernel_size = 5, stride = 2, padding = 2, bias = False))
            block_layers.append(nn.BatchNorm2d(dim))
            block_layers.append(self._make_layer(block, dim, block_layer))
            input_dim = dim

        self.residual = nn.Sequential(*block_layers)

        if pooling == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d([4, 1])
        elif pooling == 'statistic':
            self.pool = None
        else:
            raise NotImplementedError("Ohter pooling methods has been not implemented!")

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim[-1] * 4, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        for m in self.modules(): # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d): # 以2/n的开方为标准差，做均值为0的正态分布
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d): # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(planes, planes, stride)]
        in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes, planes))
        return nn.Sequential(*layers)

    def extract_embedding(self, x):
        assert len(x.size()) == 3, "the shape of input must be 3 dimensions"
        x = x.unsqueeze(1)
        x = self.residual(x)
        pooling_out = self.pool(x)
        x = pooling_out.view(pooling_out.size(0), -1)
        x = self.fc(x)
        return x, pooling_out

    def forward(self, x):
        '''
        params:
            x: input feature, B, C, T
        return:
            output of unnormalized speaker embedding
        '''
        x, _ = self.extract_embedding(x)
        return x

if __name__ == '__main__':
    import yaml
    from yaml import CLoader
    from torchsummary import summary
    f = open('./conf/config.yaml', 'r')
    opts = yaml.load(f, Loader = CLoader)
    f.close()
    print(opts['model']['resnet'])
    net = SpeakerEmbNet(opts['model'])
    print(net)
    summary(net.cuda(), (1, 257, 100))
