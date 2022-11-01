from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.nn as nn
from timm.models import swin_base_patch4_window7_224, swin_tiny_patch4_window7_224
from fastai.callback.hook import hook_outputs
from fastai.layers import ConvLayer, BatchNorm, PixelShuffle_ICNR, SelfAttention
from fastai.torch_core import apply_init

class MS_CAM(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(MS_CAM, self).__init__()
        mid_channel = channel // ratio
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        g_x = self.global_att(x)
        l_x = self.local_att(x)
        w = self.sigmoid(l_x * g_x.expand_as(l_x))
        return w * x

class AFF(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(AFF, self).__init__()
        self.ms_can = MS_CAM(channel, ratio)

    def forward(self, x0, x1):
        x0_copy = x0.clone()
        x1_copy = x1.clone()
        x01_sum = x0_copy + x1_copy
        x01_sum = self.ms_can(x01_sum)
        x0 = x01_sum * x0
        x1 = (1 - x01_sum) * x1
        return x0 + x1

class IAFF(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(IAFF, self).__init__()
        self.ms_can0 = MS_CAM(channel, ratio)
        self.ms_can1 = MS_CAM(channel, ratio)

    def forward(self, x0, x1):
        x0_copy0 = x0.clone()
        x1_copy0 = x1.clone()
        x0_copy1 = x0.clone()
        x1_copy1 = x1.clone()
        x01_sum = x0_copy0 + x1_copy0
        x01_sum = self.ms_can0(x01_sum)
        x0 = x01_sum * x0
        x1 = (1 - x01_sum) * x1
        x01_sum = x0 + x1
        x01_sum = self.ms_can1(x01_sum)
        x0 = x01_sum * x0_copy1
        x1 = (1 - x01_sum) * x1_copy1
        return x0 + x1

class UNetBlock(nn.Module):
    def __init__(self, up_in_c, x_in_c, final_div=True, blur=False, act_cls=nn.ReLU,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, norm_type=None)
        self.bn = BatchNorm(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(nf, nf, act_cls=act_cls, norm_type=norm_type,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.act = act_cls()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    def forward(self, up_in, s):
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.act(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))

class LTCrowd(nn.Module):
    def __init__(self, pretrained=True, blur=False, act_cls=nn.ReLU, self_attention=False, init=nn.init.kaiming_normal_, norm_type=None):
        super(LTCrowd, self).__init__()

        self.backbone0 = nn.Sequential(*list(swin_tiny_patch4_window7_224(pretrained=pretrained).children())[:-2])
        self.backbone1 = nn.Sequential(*list(swin_tiny_patch4_window7_224(pretrained=pretrained).children())[:-2])

        h_layer0 = []
        for name,layer in self.backbone0[2].named_modules():
            if 'mlp.fc2' in name and isinstance(layer,torch.nn.Linear):
                h_layer0.append(layer)
        self.hooks0 = hook_outputs([l for i,l in enumerate(h_layer0) if i in [1,3,9,11]],detach=False)

        h_layer1 = []
        for name,layer in self.backbone1[2].named_modules():
            if 'mlp.fc2' in name and isinstance(layer,torch.nn.Linear):
                h_layer1.append(layer)
        self.hooks1 = hook_outputs([l for i,l in enumerate(h_layer1) if i in [1,3,9,11]],detach=False)

        self.affs = nn.ModuleList([AFF(96), AFF(192), AFF(384), AFF(768)])
        self.aff_final = AFF(768)
        # self.affs = [AFF(128), AFF(256), AFF(512), AFF(1024)]
        # self.aff_final = AFF(1024)

        self.mid_conv = nn.Sequential(
            ConvLayer(768, 1024, act_cls=act_cls, norm_type=norm_type),
            ConvLayer(1024, 768, act_cls=act_cls, norm_type=norm_type)
        )

        decoder_channels = [(768, 768), (1152, 384), (960, 192), (672, 96)]
        # decoder_channels = [(1024, 1024), (1536, 512), (1280, 256), (896, 128)]
        decoder = []
        for (up_in_c, x_in_c) in decoder_channels:
            decoder.append(UNetBlock(up_in_c, x_in_c, 
                blur=blur, self_attention=self_attention, norm_type=norm_type))
        self.decoder = nn.ModuleList(decoder)

        self.final_conv = ConvLayer(432, 3, ks=1, act_cls=act_cls, norm_type=norm_type)
        # self.final_conv = ConvLayer(576, 3, ks=1, act_cls=act_cls, norm_type=norm_type)

        # apply_init(self.mid_conv, init)
        apply_init(self.final_conv, init)

    def feat_reshape(self, x):
        b, p, c = x.shape
        w = int(sqrt(p))
        assert w * w == p, 'Not square feature map'
        return x.permute(0,2,1).view(b, c, w, w)

    # x: B x 2 x C x H x W
    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]

        x0 = self.backbone0(x0)
        x1 = self.backbone1(x1)

        features = []
        for i in range(len(self.hooks0)):
            f0, f1 = self.hooks0[i].stored, self.hooks1[i].stored
            f0, f1 = self.feat_reshape(f0), self.feat_reshape(f1)
            f_merged = self.affs[i](f0, f1)
            # print(f'f{i}: {f_merged.shape}')
            features.append(f_merged)

        x0, x1 = self.feat_reshape(x0), self.feat_reshape(x1)
        x = self.aff_final(x0, x1)
        x = self.mid_conv(x)

        # print(f'x: {x.shape}')

        for i in range(len(features)):
            x = self.decoder[i](x, features[-i-1])
            # print(f'decoder{i}: {x.shape}')

        # print(f'out: {x.shape}')

        x = self.final_conv(x)

        return x
            

if __name__ == '__main__':
    def num_params(model):
        return sum([p.numel() for p in model.parameters()]) / 1e6
    model = LTCrowd()
    print(f'Num params: {num_params(model):.2f}M')
    x = torch.randn(2, 2, 3, 224, 224)
    y = model(x)
    print(y.shape)