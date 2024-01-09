import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from ..builder import BACKBONES
import logging
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

__all__ = ['MSAC1']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class MultiscaleChannel1(nn.Module):
    def __init__(self, channels, reduction):
        super(MultiscaleChannel1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels * 4, channels * 4 // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels * 4 // reduction, channels * 4, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
        self.reconv1 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.reconv2 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.reconv3 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.reconv4 = nn.Conv2d(2048, 256, kernel_size=1, bias=False)

    def forward(self, x):

        if x[-1].size()[1] != 256:
            x[0] = self.reconv1(x[0])
            x[1] = self.reconv2(x[1])
            x[2] = self.reconv3(x[2])
            x[3] = self.reconv4(x[3])
        outputs = []
        # c = x[3]
        # x = x[:3]
        module_input = x.copy()
        # for i, conv in enumerate(module_input):
        #     print('input x{}:{}'.format(i,conv.size())) #[4, 4, 256, h, w]
        for i, j in enumerate(module_input):
            x[i] = self.avg_pool(x[i])
        # for i, conv in enumerate(x):
        #     print('avgpool x{}:{}'.format(i, conv.size()))#[4, 4, 256, 1, 1]
        x = torch.cat(x, dim=1)
        # print('concact',x.size())#[4, 1024, 1, 1]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = torch.split(x, split_size_or_sections=256, dim=1)
        # for i, conv in enumerate(x):
        #     print('split x{}:{}'.format(i, conv.size()))#[4, 4, 256, 1, 1]
        for i, att in enumerate(x):
            module_input[i] = module_input[i] * att
            outputs.append(module_input[i])

        # outputs.append(c)
        # print('output', type(output[0]))
        return outputs

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

bottleneck_dic = {
    'ResNet50': Bottleneck
}

@BACKBONES.register_module()
class MSAC1(nn.Module):

    def __init__(self, block, layers,
                 reduction=16,
                 zero_init_residual=True,
                 stack_times=1,
                 num_classes=4):
        """
       Parameters
       ----------
       block (nn.Module): Bottleneck class.
           - For SENet154: SEBottleneck
           - For SE-ResNet models: SEResNetBottleneck
           - For SE-ResNeXt models:  SEResNeXtBottleneck
       layers (list of ints): Number of residual blocks for 4 layers of the
           network (layer1...layer4).
       num_classes (int): Number of outputs in `last_linear` layer.
           - For all models: 1000
       """
        self.inplanes = 64
        super(MSAC1, self).__init__()
        block = bottleneck_dic[block]
        self.zero_init_residual = zero_init_residual
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.stack_times = stack_times
        self.ms = MultiscaleChannel1(channels=256, reduction=reduction)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                if self.zero_init_residual:
                    for m in self.modules():
                        if isinstance(m, Bottleneck):
                            constant_init(m.bn3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def extraction(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)

        feature = self.ms(outputs)
        for i in range(self.stack_times-1):
            feature = self.ms(feature)
        return feature

    def forward(self, x):
        feature = self.extraction(x)
        return feature


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))


