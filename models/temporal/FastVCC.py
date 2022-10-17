import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# if two parts are jointly trained, use fastVCC.
class FastVCC(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, len_frames=3):
        super(FastVCC, self).__init__()
        self.len_frames = len_frames
        self.lcn = LCN(in_channels=3, out_channels=1)
        self.drbs = DRBs(num_stages=3, num_layers=3, num_f_maps=5, in_channels=self.len_frames, out_channels=self.len_frames)
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: input of multiple frames
                  (N, F, C, Hin, Win), N=batch size, F= the length of frames
        :return:
                lcn_outputs: output from LCN, (N, F, Cout, Hout, Wout) Cout=1, Hout=Hin/8, Wout=Win/8
                count_outputs: output from counting layer of each DRB block,
                               (num_stages, N, Cout, Hout, Wout), Cout=1, Hout=Hin/8, Wout=Win/8

        """
        assert self.len_frames == x.shape[1]
        for i in range(self.len_frames):
            lcn_out = self.lcn(x[:, i, :, :, :])  # (N, Cout, Hout, Wout)
            if i == 0:
                lcn_outputs = lcn_out.unsqueeze(1)
            else:
                lcn_outputs = torch.cat((lcn_outputs, lcn_out.unsqueeze(1)), dim=1)     # (N, F, Cout, Hout, Wout)

        # reshape and concatenate
        drbs_input = torch.reshape(lcn_outputs, (lcn_outputs.shape[0], lcn_outputs.shape[1], -1))   # (N, F, Cout*Hout*Wout) Cout=1
        drbs_output = self.drbs(drbs_input)     # (num_stages, N, F, Hout*Wout)

        # normalization to get weights
        l1norm = torch.sum(torch.abs(drbs_output), dim=-1)     # (num_stages, N, F)
        weights = l1norm/l1norm.sum(dim=-1, keepdim=True)     # (num_stages, N, F)

        # counting layer
        count_outputs = lcn_outputs.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (num_stages, N, Cout, Hout, Wout)

        return lcn_outputs, count_outputs

class LCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, load_weights=False):
        super(LCN, self).__init__()
        self.channels_cfg = [8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 16, 8]
        self.lcn = make_layers(self.channels_cfg, in_channels)
        self.output_layer = nn.Conv2d(8, out_channels, kernel_size=1)
        if not load_weights:
            self._initialize_weights()

    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: input: a single frame, (N, Cin, H, W)
        :return: x: the output size is 1/8 of original input size, (N, Cout, Hout, Wout)
        """
        x = self.lcn(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DRBs(nn.Module):
    # def __init__(self, num_stages=3, num_layers=3, num_f_maps=5, in_channels=5, out_channels=5, load_weights=False):
    def __init__(self, num_stages=3, num_layers=3, num_f_maps=5, in_channels=5, out_channels=5):
        """
        :param num_stages: the number of DRB block, default 3
        :param num_layers: the number of DilatedResidualLayer, default 3
        :param num_f_maps: the number of medium feature_maps
        :param in_channels: should be the length of frames per input, default 5
        :param out_channels: should be equal to in_channels, default 5
        """
        super(DRBs, self).__init__()
        self.stage1 = DRB(num_layers, num_f_maps, in_channels, out_channels)
        self.stages = nn.ModuleList([copy.deepcopy(DRB(num_layers, num_f_maps, out_channels, out_channels)) for s in range(num_stages-1)])
        # if not load_weights:
        #     self._initialize_weights()

    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: tensor after reshape and concatenation of LCN outputs
                  (N, F, H*W), F represent the length of frames
        :return: outputs of each DRB stage
                  (num_stages, N, F, H*W)
        """
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


class DRB(nn.Module):
    def __init__(self, num_layers, num_f_maps, in_channels, out_channels):
        super(DRB, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, out_channels, 1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.dropout = nn.Dropout()

    @torch.cuda.amp.autocast()
    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        # out = self.dropout(out)   # TODO: use dropout?
        return (x + out)


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
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

if __name__ == '__main__':
    m = FastVCC(len_frames=5)
    x = torch.randn(1, 5, 3, 256, 256)
    y = m(x)
    print(y.shape)