# Pytorch implementation of STDNet
# Reference: https://github.com/stk513486/STDNet/blob/master/Train.py

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import models

from collections import OrderedDict

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class ResizeAndAdjustChannel(nn.Module):
    def __init__(self, resize_shape, in_channels, out_channels):
        super(ResizeAndAdjustChannel, self).__init__()
        self.resize_shape = resize_shape
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.resize(x, self.resize_shape)
        x = self.conv(x)
        return x

class OutputAdjust(nn.Module):
    def __init__(self, resize_shape, in_channels, out_channels):
        super(OutputAdjust, self).__init__()
        self.resize_shape = resize_shape
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.resize(x, self.resize_shape)
        return x

class CommonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CommonConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class DilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class CommonConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CommonConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class DilatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class DenseSpatialBlock(nn.Module):
    def __init__(self):
        super(DenseSpatialBlock, self).__init__()
        self.channel_list = [64, 64, 64]
        self.cconv1 = CommonConv2d(512, 256)
        self.dconv1 = DilatedConv2d(256, self.channel_list[0], 1)
        self.cconv2 = CommonConv2d(512+self.channel_list[0], 256)
        self.dconv2 = DilatedConv2d(256, self.channel_list[1], 2)
        self.cconv3 = CommonConv2d(512+self.channel_list[0]+self.channel_list[1], 256)
        self.dconv3 = DilatedConv2d(256, self.channel_list[2], 3)
        self.cconv4 = CommonConv2d(512+self.channel_list[0]+self.channel_list[1]+self.channel_list[2], 512)

    def forward(self, x):
        z1 = self.cconv1(x)
        z1 = self.dconv1(z1)
        z2 = torch.cat((x, z1), dim=1)

        z2 = self.cconv2(z2)
        z2 = self.dconv2(z2)
        z3 = torch.cat((x, z1, z2), dim=1)

        z3 = self.cconv3(z3)
        z3 = self.dconv3(z3)
        z4 = torch.cat((x, z1, z2, z3), dim=1)

        z4 = self.cconv4(z4)
        return z4

class SpatialChannelAwareBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialChannelAwareBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(in_channels, 256)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.gap(x)
        a = a.view(a.size(0), a.size(1))
        a = self.dense1(a)
        a = self.relu(a)
        a = self.dense2(a)
        a = self.sigmoid(a)
        a = a.view(a.size(0), a.size(1), 1, 1)
        x = x * a
        return x

class DenseTemporalBlock(nn.Module):
    def __init__(self):
        super(DenseTemporalBlock, self).__init__()
        self.channel_list = [64, 64, 64]
        self.cconv1 = CommonConv3d(512, 256)
        self.dconv1 = DilatedConv3d(256, self.channel_list[0], 1)
        self.cconv2 = CommonConv3d(512+self.channel_list[0], 256)
        self.dconv2 = DilatedConv3d(256, self.channel_list[1], 2)
        self.cconv3 = CommonConv3d(512+self.channel_list[0]+self.channel_list[1], 256)
        self.dconv3 = DilatedConv3d(256, self.channel_list[2], 3)
        self.cconv4 = CommonConv3d(512+self.channel_list[0]+self.channel_list[1]+self.channel_list[2], 512)

    def forward(self, x):
        z1 = self.cconv1(x)
        z1 = self.dconv1(z1)
        z2 = torch.cat((x, z1), dim=1)

        z2 = self.cconv2(z2)
        z2 = self.dconv2(z2)
        z3 = torch.cat((x, z1, z2), dim=1)

        z3 = self.cconv3(z3)
        z3 = self.dconv3(z3)
        z4 = torch.cat((x, z1, z2, z3), dim=1)

        z4 = self.cconv4(z4)
        return z4

class TemporalChannelAwareBlock(nn.Module):
    def __init__(self, in_channels):
        super(TemporalChannelAwareBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dense1 = nn.Linear(in_channels, 128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.gap(x)
        a = a.view(a.size(0), a.size(1))
        a = self.dense1(a)
        a = self.relu(a)
        a = self.dense2(a)
        a = self.sigmoid(a)
        a = a.view(a.size(0), a.size(1), 1, 1, 1)
        x = x * a
        return x

class SpatialBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialBlock, self).__init__()
        self.dense = DenseSpatialBlock()
        self.ca = SpatialChannelAwareBlock(in_channels)

    def forward(self, x):
        x = self.dense(x)
        x = self.ca(x)
        return x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels):
        super(TemporalBlock, self).__init__()
        self.dense = DenseTemporalBlock()
        self.ca = TemporalChannelAwareBlock(in_channels)

    def forward(self, x):
        x = self.dense(x)
        x = self.ca(x)
        return x

class STDNet(nn.Module):
    def __init__(self, resize_shape, in_channels, load_weights):
        super(STDNet, self).__init__()
        output_shape = (resize_shape[0] // 2, resize_shape[1] // 2)
        self.resize_and_adjust = ResizeAndAdjustChannel(resize_shape, in_channels, 3)
        self.vgg_features = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.vgg = make_layers(self.vgg_features)
        self.sb1 = SpatialBlock(512)
        self.tb1 = TemporalBlock(512)
        self.sb2 = SpatialBlock(512)
        self.tb2 = TemporalBlock(512)
        self.sb3 = SpatialBlock(512)
        self.tb3 = TemporalBlock(512)
        self.sb4 = SpatialBlock(512)
        self.tb4 = TemporalBlock(512)
        self.dconv1 = DilatedConv2d(512, 128, 1)
        self.dconv2 = DilatedConv2d(128, 64, 1)
        self.output_adjust = OutputAdjust(output_shape, 64, 1)

        if load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._initialize_weights()
            fsd = OrderedDict()
            # 10 convlution *(weight, bias) = 20 parameters
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.vgg.load_state_dict(fsd)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b*t, c, h, w)
        x = self.resize_and_adjust(x)
        x = self.vgg(x)

        x = self.sb1(x)
        x = x.view(b, x.size(1), t, x.size(2), x.size(3))
        x = self.tb1(x)
        x = x.view(b*t, x.size(1), x.size(3), x.size(4))

        x = self.sb2(x)
        x = x.view(b, x.size(1), t, x.size(2), x.size(3))
        x = self.tb2(x)
        x = x.view(b*t, x.size(1), x.size(3), x.size(4))

        x = self.sb3(x)
        x = x.view(b, x.size(1), t, x.size(2), x.size(3))
        x = self.tb3(x)
        x = x.view(b*t, x.size(1), x.size(3), x.size(4))

        x = self.sb4(x)
        x = x.view(b, x.size(1), t, x.size(2), x.size(3))
        x = self.tb4(x)
        x = x.view(b*t, x.size(1), x.size(3), x.size(4))

        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.output_adjust(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3))

        return x
        
if __name__ == '__main__':
    model = STDNet((256, 256), 3, False)
    x = torch.randn(2, 8, 3, 512, 512)
    y = model(x)
    print(y.size())