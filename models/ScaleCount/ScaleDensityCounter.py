import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops.layers.torch import Rearrange

import collections

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# the module definition for the multi-branch in the density head
class MultiBranchModule(nn.Module):
    def __init__(self, in_channels, sync=False):
        super(MultiBranchModule, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch1x1_1 = BasicConv2d(in_channels//2, in_channels, kernel_size=1, sync=sync)

        self.branch3x3_1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch3x3_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=(3, 3), padding=(1, 1), sync=sync)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch3x3dbl_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=5, padding=2, sync=sync)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.branch1x1_1(branch1x1)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        outputs = [branch1x1, branch3x3, branch3x3dbl, x]
        return torch.cat(outputs, 1)

# the module definition for the basic conv module
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, sync=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        if sync:
            # for sync bn
            print('use sync inception')
            self.bn = nn.SyncBatchNorm(out_channels, eps=0.001)
        else:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Decoder, self).__init__()
        self.conv1 = Conv2d(in_dim, hid_dim, 3, same_padding=True, NL='relu')
        self.conv2 = Conv2d(hid_dim, out_dim, 3, same_padding=True, NL='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DensityHead(nn.Module):
    def __init__(self, in_dim):
        super(DensityHead, self).__init__()
        self.mbm = MultiBranchModule(in_dim)
        self.conv = Conv2d(in_dim*4, 1, 1, same_padding=True)

    def forward(self, x):
        x = self.mbm(x)
        x = self.conv(x)
        return x

class DensityHead2(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(DensityHead2, self).__init__()
        self.conv1 = Conv2d(in_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv2 = Conv2d(hid_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv3 = Conv2d(hid_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv4 = Conv2d(hid_dim, 64, 3, dilation=2, NL='relu')
        self.conv5 = Conv2d(64, 1, 1, same_padding=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class ScaleHead(nn.Module):
    def __init__(self, in_dim):
        super(ScaleHead, self).__init__()
        self.mbm = MultiBranchModule(in_dim)
        self.conv = Conv2d(in_dim*4, 1, 1, same_padding=True)
        self.pool = nn.AvgPool2d(16)

    def forward(self, x):
        x = self.mbm(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class ScaleHead2(nn.Module):
    def __init__(self, in_dim):
        super(ScaleHead2, self).__init__()
        self.conv1 = Conv2d(in_dim, 512, 3, dilation=2, NL='relu')
        self.conv2 = Conv2d(512, 512, 3, dilation=2, NL='relu')
        self.conv3 = Conv2d(512, 512, 3, dilation=2, NL='relu')
        self.conv4 = Conv2d(512, 256, 3, dilation=2, NL='relu')
        self.conv5 = Conv2d(256, 128, 3, dilation=2, NL='relu')
        self.conv6 = Conv2d(128, 64, 3, dilation=2, NL='relu')
        self.conv7 = Conv2d(64, 1, 1, same_padding=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x

class SDCNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SDCNet, self).__init__()
        
        # define the backbone network
        if pretrained:
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        else:
            vgg = models.vgg16_bn()

        features = list(vgg.features.children())
        # get each stage of the backbone
        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])
        self.features5 = nn.Sequential(*features[33:43])

        # decoder definition
        self.de_pred5 = Decoder(512, 1024, 512)
        self.de_pred4 = Decoder(1024, 512, 256)
        self.de_pred3 = Decoder(512, 256, 128)
        self.de_pred2 = Decoder(256, 128, 64)
        self.de_pred1 = Decoder(128, 64, 64)

        # density head definition
        # self.density_head5 = DensityHead(512)
        # self.density_head4 = DensityHead(256)
        # self.density_head3 = DensityHead(128)
        # self.density_head2 = DensityHead(64)
        # self.density_head1 = DensityHead(64)

        self.density_head5 = DensityHead2(512, 512)
        self.density_head4 = DensityHead2(256, 512)
        self.density_head3 = DensityHead2(128, 256)
        self.density_head2 = DensityHead2(64, 128)
        self.density_head1 = DensityHead2(64, 128)

        # scale definition
        self.scale_decoder = Decoder(512, 1024, 512)
        # self.scale_head = ScaleHead(64)
        self.scale_head = DensityHead2(512, 512)
        
    def forward(self, x):
        # get the output of each stage
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)

        # begining of decoding
        x = self.de_pred5(x5)
        x5_out = x
        x = F.upsample_bilinear(x, size=x4.size()[2:])

        x = torch.cat([x4, x], 1)
        x = self.de_pred4(x)
        x4_out = x
        x = F.upsample_bilinear(x, size=x3.size()[2:])

        x = torch.cat([x3, x], 1)
        x = self.de_pred3(x)
        x3_out = x
        x = F.upsample_bilinear(x, size=x2.size()[2:])

        x = torch.cat([x2, x], 1)
        x = self.de_pred2(x)
        x2_out = x
        x = F.upsample_bilinear(x, size=x1.size()[2:])

        x = torch.cat([x1, x], 1)
        x = self.de_pred1(x)
        x1_out = x

        # density prediction
        x5_density = self.density_head5(x5_out)
        x4_density = self.density_head4(x4_out)
        x3_density = self.density_head3(x3_out)
        x2_density = self.density_head2(x2_out)
        x1_density = self.density_head1(x1_out)

        b, _, h, w = x5_density.size()

        x1_density = x1_density.reshape([b, 1, h, 16, w, 16]).sum(dim=(3, 5))
        x2_density = x2_density.reshape([b, 1, h, 8, w, 8]).sum(dim=(3, 5))
        x3_density = x3_density.reshape([b, 1, h, 4, w, 4]).sum(dim=(3, 5))
        x4_density = x4_density.reshape([b, 1, h, 2, w, 2]).sum(dim=(3, 5))

        densities = torch.cat([x1_density, x2_density, x3_density, x4_density, x5_density], 1)

        # scale prediction
        x5_out2 = self.scale_decoder(x5)
        # print(x5_out2.max(), x5_out2.min())
        scale = self.scale_head(x5_out2)

        # # scale fusion
        # with torch.no_grad():
        #     lower_scale = torch.clamp(torch.floor(scale).to(torch.long), 0, 4)
        #     # upper_scale = torch.clamp(torch.ceil(scale).to(torch.long), 0, 4)
        #     # lower_weight = upper_scale - scale
        #     # upper_weight = scale - lower_scale
        #     lower_scale = F.one_hot(lower_scale, 5).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
        #     # upper_scale = F.one_hot(upper_scale, 5).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)

        # # density = (lower_weight * lower_scale + upper_weight * upper_scale) * densities
        # density = lower_scale * densities
        # density = density.sum(dim=1, keepdim=True)
        
        return densities, scale

if __name__ == '__main__':
    m = SDCNet()
    x = torch.randn(2, 3, 512, 512)
    den, scale = m(x)
    print(den.shape, scale.shape)
    # print(den.max(), den.min(), den.mean())
    print(scale.max(), scale.min(), scale.mean())
    
