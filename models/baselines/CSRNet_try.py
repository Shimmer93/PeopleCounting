# credit: https://github.com/CommissarMa/CSRNet-pytorch/blob/master/model.py
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from einops import rearrange
from involution import Involution2d

import collections

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    @torch.cuda.amp.autocast()
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    @torch.cuda.amp.autocast()
    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        if load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            # 10 convlution *(weight, bias) = 20 parameters
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        # x = self.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

class CSRInvoNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRInvoNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_invo_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        if load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            # 10 convlution *(weight, bias) = 20 parameters
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        x = self.frontend(x)
        print(torch.max(x))
        x = self.backend(x)
        x = self.output_layer(x)
        print(torch.max(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_invo_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Involution2d(in_channels, v)
            layers += [conv2d]
            # if batch_norm:
            #     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            # else:
            #     layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)




def make_layers_temporal(cfg, in_channels=3, frames=8, batch_norm=False, dilation=False):
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
                layers += [conv2d, Aggregate(frames, v), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, Aggregate(frames, v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class TemporalAggregate(nn.Module):
    def __init__(self):
        super(TemporalAggregate, self).__init__()
        self.blur = GaussianBlur(7.5, 1.5)
        
    # x: B x T x C x H x W
    def forward(self, x):
        b = x.shape[0]
        x_blur = rearrange(x, 'b t c h w -> (b c) t h w')
        x_blur = self.blur(x_blur)
        x_blur_prev = x_blur[:, :-1, ...]
        x_blur_next = x_blur[:, 1:, ...]
        x_blur_prev = F.pad(x_blur_prev, (0, 0, 0, 0, 1, 0))
        x_blur_next = F.pad(x_blur_next, (0, 0, 0, 0, 0, 1))
        x_aggregate = x_blur_prev + x_blur_next
        x_aggregate = rearrange(x_aggregate, '(b c) t h w -> b t c h w', b=b)
        x_aggregate = x + x_aggregate
        return x_aggregate

class TemporalWeightedAggregate(nn.Module):
    def __init__(self, dim):
        super(TemporalWeightedAggregate, self).__init__()
        self.blur = GaussianBlur(7, 1.5)
        self.atten = nn.Conv2d(3, 1, kernel_size=1)
        self.atten.weight.data.fill_(0.5)
        self.atten.bias.data.fill_(0.5)
        self.post_conv = nn.Conv3d(dim, dim, kernel_size=1)

    # x: B x T x C x H x W
    def forward(self, x):
        b, c = x.shape[0], x.shape[2]
        x_blur = rearrange(x, 'b t c h w -> (b c) t h w')
        x_blur = self.blur(x_blur)
        x_blur_prev = x_blur[:, :-1, ...]
        x_blur_next = x_blur[:, 1:, ...]
        x_blur_prev = F.pad(x_blur_prev, (0, 0, 0, 0, 1, 0))
        x_blur_next = F.pad(x_blur_next, (0, 0, 0, 0, 0, 1))
        x_reshape = rearrange(x, 'b t c h w -> (b c) t h w')
        x_agg = torch.stack([x_reshape, x_blur_prev, x_blur_next], dim=2)
        x_agg = rearrange(x_agg, 'bc t d h w -> (bc t) d h w')
        x_agg = self.atten(x_agg)
        x_agg = x_agg.squeeze(1)
        x_agg = rearrange(x_agg, '(b c t) h w -> b c t h w', b=b, c=c)
        x_agg = self.post_conv(x_agg)
        x_agg = rearrange(x_agg, 'b c t h w -> b t c h w')
        return x_agg

class Aggregate(nn.Module):
    def __init__(self, frames, dim):
        super(Aggregate, self).__init__()
        self.ta = TemporalWeightedAggregate(dim)
        self.t = frames

    # x: (B x T) x C x H x W
    def forward(self, x):
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.t)
        x = self.ta(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        return x

class TemporalCSRNet(CSRNet):
    def __init__(self, load_weights=False, frames=8):
        super(TemporalCSRNet, self).__init__(load_weights)
        self.backend = make_layers_temporal(
            self.backend_feat, in_channels=512, frames=frames, dilation=True)
        
    # x: B x T x C x H x W
    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        return x

if __name__ == '__main__':
    m = CSRInvoNet()
    x = torch.randn(2, 3, 512, 512)
    y = m(x)
    print(y.shape)
    print(y[0, 0, 0, 0])