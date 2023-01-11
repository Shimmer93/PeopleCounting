import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

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
    def __init__(self, in_dim, hid_dim):
        super(DensityHead, self).__init__()
        self.conv1 = Conv2d(in_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv2 = Conv2d(hid_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv3 = Conv2d(hid_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv4 = Conv2d(hid_dim, in_dim, 3, dilation=2, NL='relu')
        self.conv5 = Conv2d(in_dim, 64, 3, dilation=2, NL='relu')
        self.conv6 = Conv2d(64, 1, 1, same_padding=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class ScaleHead(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(ScaleHead, self).__init__()
        self.conv1 = Conv2d(in_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv2 = Conv2d(hid_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv3 = Conv2d(hid_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv4 = Conv2d(hid_dim, 256, 3, dilation=2, NL='relu')
        self.conv5 = Conv2d(256, 64, 3, dilation=2, NL='relu')
        self.conv6 = Conv2d(64, 1, 1, same_padding=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class MDBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MDBlock, self).__init__()
        self.conv1 = Conv2d(in_dim, hid_dim, 3, same_padding=True, NL='relu')
        self.conv2 = Conv2d(in_dim, hid_dim, 3, dilation=2, NL='relu')
        self.conv3 = Conv2d(in_dim, hid_dim, 3, dilation=3, NL='relu')
        self.conv4 = Conv2d(in_dim, out_dim, 1, NL='relu')
        self.conv5 = Conv2d(hid_dim*3, out_dim, 1, NL='relu')
        self.head = nn.Sequential(
            Conv2d(out_dim, 64, 3, same_padding=True, NL='relu'),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv5(x)
        x = x + x4
        return x

class MDDensityHead(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(MDDensityHead, self).__init__()
        self.conv1 = MDBlock(in_dim, hid_dim, hid_dim)
        self.conv3 = MDBlock(hid_dim, hid_dim//2, hid_dim)
        self.conv4 = MDBlock(hid_dim, hid_dim//2, hid_dim)
        self.conv5 = Conv2d(hid_dim, 64, 3, dilation=2, NL='relu')
        self.conv6 = Conv2d(64, 1, 1, same_padding=True)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class SDCNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SDCNet, self).__init__()
        
        # define the backbone network
        if pretrained:
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        else:
            vgg = models.vgg16_bn()

        self.teacherforcing = True

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
        # self.de_pred2 = Decoder(256, 128, 64)
        # self.de_pred1 = Decoder(128, 64, 64)

        self.density_head5 = DensityHead(512, 512)
        self.density_head4 = DensityHead(256, 512)
        self.density_head3 = DensityHead(128, 256)
        # self.density_head2 = DensityHead(64, 128)
        # self.density_head1 = DensityHead(64, 128)

        # scale definition
        # self.scale_decoder = Decoder(64, 128, 128)
        # self.scale_head = ScaleHead(128, 256)
        self.scale_decoder = Decoder(512, 1024, 512)
        self.scale_head = ScaleHead(512, 512)

        
    def forward(self, x):
        if self.training:
            img, gt_scale = x
            densities, scale = self.forward_image(img)
            with torch.no_grad():
                if self.teacherforcing == True:
                    lower_scale = torch.clamp(torch.floor(gt_scale).to(torch.long), 5, 7) - 5
                else:
                    lower_scale = torch.clamp(torch.floor(scale/10).to(torch.long), 5, 7) - 5
                lower_scale = F.one_hot(lower_scale, 3).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)

        else:
            densities, scale = self.forward_image(x)
            with torch.no_grad():
                lower_scale = torch.clamp(torch.floor(scale/10).to(torch.long), 5, 7) - 5
                lower_scale = F.one_hot(lower_scale, 3).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)

        if not self.training:
            lower_scale[:,0, ...] *= 1
            lower_scale[:,1, ...] *= 1/4
            lower_scale[:,2, ...] *= 1/16

        density = lower_scale * densities
        density = density.sum(dim=1, keepdim=True)

        return density, scale, densities

    def forward_image(self, x):
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

        # x = torch.cat([x2, x], 1)
        # x = self.de_pred2(x)
        # x2_out = x
        # x = F.upsample_bilinear(x, size=x1.size()[2:])

        # x = torch.cat([x1, x], 1)
        # x = self.de_pred1(x)
        # x1_out = x

        # print(f'x1: {x1_out.max()}, x2: {x2_out.max()}, x3: {x3_out.max()}, x4: {x4_out.max()}, x5: {x5_out.max()}')

        # density prediction
        x5_density = self.density_head5(x5_out)
        x4_density = self.density_head4(x4_out)
        x3_density = self.density_head3(x3_out)
        # x2_density = self.density_head2(x2_out)
        # x1_density = self.density_head1(x1_out)

        # b, _, h, w = x5_density.size()

        x3_density = F.interpolate(x3_density, scale_factor=1/4, mode='bilinear', align_corners=True)
        x4_density = F.interpolate(x4_density, scale_factor=1/2, mode='bilinear', align_corners=True)

        # densities = torch.cat([x1_density, x2_density, x3_density, x4_density, x5_density], 1)
        densities = torch.cat([x3_density, x4_density, x5_density], 1)

        # scale prediction
        x5_out2 = self.scale_decoder(x5)
        scale = self.scale_head(x5_out2)
        
        return densities, scale

if __name__ == '__main__':
    m = SDCNet(pretrained=True)
    m.eval()
    x = torch.randn(2, 3, 512, 512)
    den, scale = m(x)
    print(den.shape, scale.shape)
    # print(den.max(), den.min(), den.mean())
    print(scale.max(), scale.min(), scale.mean())
    # print(dens[:, 0].max(), dens[:, 1].max(), dens[:, 2].max(), dens[:, 3].max(), dens[:, 4].max())