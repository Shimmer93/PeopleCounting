import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from fastai.layers import PixelShuffle_ICNR

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x) 
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Head(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.stage1 = ConvBlock(in_dim, in_dim//2)
        self.stage2 = ConvBlock(in_dim, in_dim//2, padding=2, dilation=2)
        self.stage3 = ConvBlock(in_dim, in_dim//2, padding=3, dilation=3)
        self.stage4 = ConvBlock(in_dim, in_dim*3//2, kernel_size=1, padding=0)
        self.res = nn.Sequential(
            ConvBlock(in_dim*3//2, 128, kernel_size=3, padding=1),
            ConvBlock(128, 1, kernel_size=1, padding=0, relu=False)
        )

        self._init_params()

    def forward(self, x):
        y1 = self.stage1(x)
        y2 = self.stage2(x)
        y3 = self.stage3(x)
        y4 = self.stage4(x)
        y = torch.cat((y1,y2,y3), dim=1) + y4
        y = self.res(y)
        return y

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SwinSDCNet3(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.teacherforcing = True

        if pretrained:
            swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
            # vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        else:
            swin = models.swin_b(weights=None)
            # vgg = models.vgg16_bn(weights=None)

        features = list(swin.features.children())

        # features_vgg = list(vgg.features.children())
        # self.feat_vgg = nn.Sequential(
        #     *features_vgg[0:23]
        #     # ConvBlock(256, 128, kernel_size=1, padding=0, relu=False)
        # )

        self.feat1 = nn.Sequential(*features[0:2])
        self.feat2 = nn.Sequential(*features[2:4])
        self.feat3 = nn.Sequential(*features[4:6])
        self.feat4 = nn.Sequential(*features[6:8])

        self.dec4 = nn.Sequential(
            ConvBlock(1024, 2048),
            ConvBlock(2048, 1024)
        )

        self.dec3 = nn.Sequential(
            ConvBlock(1024, 1024),
            ConvBlock(1024, 512)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 256)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(256, 256),
            # ConvBlock(256+256, 256),
            ConvBlock(256, 128)
        )

        self.shuf4 = PixelShuffle_ICNR(1024, 512, act_cls=nn.ReLU)
        self.shuf3 = PixelShuffle_ICNR(512, 256, act_cls=nn.ReLU)
        self.shuf2 = PixelShuffle_ICNR(256, 128, act_cls=nn.ReLU)

        self.head3 = nn.Sequential(
            ConvBlock(512, 512, padding=2, dilation=2),
            ConvBlock(512, 512, padding=2, dilation=2),
            ConvBlock(512, 512, padding=2, dilation=2),
            ConvBlock(512, 256, padding=2, dilation=2),
            ConvBlock(256, 128, padding=2, dilation=2),
            ConvBlock(128, 1, kernel_size=1, padding=0, relu=False)
            # ConvBlock(512, 1, kernel_size=1, padding=0, relu=False)
        )

        self.head2 = nn.Sequential(
            ConvBlock(256, 256, padding=2, dilation=2),
            ConvBlock(256, 256, padding=2, dilation=2),
            ConvBlock(256, 256, padding=2, dilation=2),
            ConvBlock(256, 128, padding=2, dilation=2),
            ConvBlock(128, 1, kernel_size=1, padding=0, relu=False)
            # ConvBlock(256, 1, kernel_size=1, padding=0, relu=False)
        )

        self.head1 = nn.Sequential(
            ConvBlock(128, 128, padding=2, dilation=2),
            ConvBlock(128, 128, padding=2, dilation=2),
            ConvBlock(128, 128, padding=2, dilation=2),
            ConvBlock(128, 1, kernel_size=1, padding=0, relu=False)
        )

        self.scale_dec = nn.Sequential(
            ConvBlock(1024, 512, padding=2, dilation=2),
            ConvBlock(512, 512, padding=2, dilation=2),
            ConvBlock(512, 512, padding=2, dilation=2),
            ConvBlock(512, 256, padding=2, dilation=2),
            ConvBlock(256, 128, padding=2, dilation=2),
            ConvBlock(128, 1, kernel_size=1, padding=0, relu=False)
        )

    def _forward(self, x):
        x1 = self.feat1(x) # 128
        x2 = self.feat2(x1) # 256
        x3 = self.feat3(x2) # 512
        x4 = self.feat4(x3) # 1024

        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)

        # x1_vgg = self.feat_vgg(x)
        
        d4 = self.dec4(x4) # 1024
        x = self.shuf4(d4) # 512

        x = torch.cat([x, x3], dim=1) # 1024
        x_scale = x
        d3 = self.dec3(x) # 512
        o3 = d3 
        x = self.shuf3(d3) # 256

        x = torch.cat([x, x2], dim=1) # 512
        d2 = self.dec2(x) # 256
        o2 = d2
        x = self.shuf2(d2) # 128

        x = torch.cat([x, x1], dim=1) # 256
        # x = torch.cat([x, x1, x1_vgg], dim=1) # 256
        d1 = self.dec1(x) # 128
        o1 = d1

        o3 = self.head3(o3)
        o2 = self.head2(o2)
        o1 = self.head1(o1)

        o1 = F.interpolate(o1, scale_factor=1/4, mode='bilinear', align_corners=True)
        o2 = F.interpolate(o2, scale_factor=1/2, mode='bilinear', align_corners=True)
        # o2 = F.interpolate(o2, scale_factor=2, mode='bilinear', align_corners=True)
        # o3 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=True)

        dens = torch.cat([o1, o2, o3], dim=1)

        scale = self.scale_dec(x_scale)

        return dens, scale

    def forward(self, x):
        if self.training:
            img, gt_scale = x
            dens, scale = self._forward(img)
            with torch.no_grad():
                if self.teacherforcing == True:
                    lower_scale = torch.clamp(torch.floor(gt_scale).to(torch.long), 4, 6) - 4
                else:
                    lower_scale = torch.clamp(torch.floor(scale/10).to(torch.long), 4, 6) - 4
                lower_scale = F.one_hot(lower_scale, 3).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
        else:
            dens, scale = self._forward(x)
            with torch.no_grad():
                lower_scale = torch.clamp(torch.floor(scale/10).to(torch.long), 4, 6) - 4
                lower_scale = F.one_hot(lower_scale, 3).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)

        if not self.training:
            lower_scale[:,0, ...] *= 1
            lower_scale[:,1, ...] *= 1/4
            lower_scale[:,2, ...] *= 1/16

        den = lower_scale * dens
        den = den.sum(dim=1, keepdim=True)

        return den, scale, dens

if __name__ == '__main__':
    m = SwinSDCNet3()
    m.eval()
    x = torch.randn(2, 3, 32, 512)
    dens, scale, _ = m(x)
    print(dens.shape, scale.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(count_parameters(m))