import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from .model_util import *
from .seg_model import DeeplabMulti

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}
'''
Sequential blocks
'''


class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes']

        model_seg = DeeplabMulti(num_classes=self.n_classes)

        self.layer0 = nn.Sequential(model_seg.conv1, model_seg.bn1, model_seg.relu, model_seg.maxpool)
        self.layer1 = model_seg.layer1
        self.layer2 = model_seg.layer2
        self.layer3 = model_seg.layer3
        self.layer4 = model_seg.layer4

        self.final1 = model_seg.layer5
        self.final2 = model_seg.layer6

    def forward(self, x):
        #inp_shape = x.shape[2:]

        x = self.layer0(x)
        # [2, 64, 65, 129]
        x = self.layer1(x)
        x = self.layer2(x)
        shared_shallow = x
        #4*512*33*65

        x = self.layer3(x)
        pred1 = self.final1(x)

        shared_seg = self.layer4(x)
        pred2 = self.final2(shared_seg)

        return shared_shallow, pred1, pred2, shared_seg

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.final1.parameters())
        b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class Classifier(nn.Module):
    def __init__(self, inp_shape):
        super(Classifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        self.inp_shape = inp_shape

        # PSPNet_Model = PSPNet(pretrained=True)

        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)
        x = F.upsample(x, size=self.inp_shape, mode='bilinear')
        return x


class ImgDecoder(nn.Module):
    def __init__(self, bottleneck_channel):
        super(ImgDecoder, self).__init__()
        self.bottleneck_channel = bottleneck_channel

        self.main = []
        self.upsample = nn.Sequential(
            # input: 1/8 * 1/8
            nn.Upsample(scale_factor=2),
            Conv2dBlock(256, 128, 3, 1, 1, norm='in', activation='relu', pad_type='reflect'),
            # 1/4 * 1/4
            nn.Upsample(scale_factor=2),
            Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation='relu', pad_type='reflect'),
            # 1/2 * 1/2
            nn.Upsample(scale_factor=2),
            Conv2dBlock(64, 32, 3, 1, 1, norm='in', activation='relu', pad_type='reflect'),
            # 1 * 1
            Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='tanh', pad_type='reflect'))

        self.main += [Conv2dBlock(self.bottleneck_channel, 256, 3, stride=1, padding=1, norm='in', activation='relu', pad_type='reflect')]
        self.main += [ResBlocks(3, 256, 'in', 'relu', pad_type='reflect')]
        self.main += [self.upsample]

        self.main = nn.Sequential(*self.main)


    def forward(self, shared_feature):
        _, _, H, W = shared_feature.size()
        shared_feature = F.interpolate(shared_feature, size=(H-1, W-1), mode='bilinear', align_corners=True)
        output = self.main(shared_feature)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # FCN classification layer
        self.dim = 64
        self.n_layer = 4
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.num_scales = 3
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(3, dim, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


class SegDiscriminator(nn.Module):
    def __init__(self):
        super(SegDiscriminator, self).__init__()
        n_classes = pspnet_specs['n_classes']
        # FCN classification layer

        self.feature = nn.Sequential(
            Conv2dBlock(n_classes, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.feature(x)
        return x
