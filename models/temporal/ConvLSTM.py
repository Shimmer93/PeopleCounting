"""
ConvLSTM for video-based crowd counting
Reference: Spatiotemporal Modeling for Crowd Counting in Videos, ICCV2017
Link: https://openaccess.thecvf.com/content_ICCV_2017/papers/Xiong_Spatiotemporal_Modeling_for_ICCV_2017_paper.pdf
"""

import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    @torch.cuda.amp.autocast()
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        load_weights = True
        hdim=64

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=3, # original 3
                                            hidden_dim=128, #64
                                            kernel_size=(3, 3),
                                            bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=128, #64
                                            hidden_dim=hdim,
                                            kernel_size=(3, 3),
                                            bias=True)
                                        
        self.decoder_1_convlstm = ConvLSTMCell(input_dim=hdim,
                                            hidden_dim=hdim,
                                            kernel_size=(3, 3),
                                            bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=hdim,
                                            hidden_dim=hdim,
                                            kernel_size=(3, 3),
                                            bias=True)
        self.decoder_CNN = nn.Conv3d(in_channels=hdim, # hdim
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

 
    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []
        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        #outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def autoencoder2(self, x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []
        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=h_t2,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs
    
    @torch.cuda.amp.autocast()
    def forward(self,x):
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        #future_seq = 3
        outputs = self.autoencoder2(x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        
        #outputs = torch.stack(x, 1)
        #outputs = x.permute(0, 2, 1, 3, 4)
        #outputs = self.decoder_CNN(outputs)

        #outputs = F.upsample(outputs, scale_factor=8)
#         output1 = F.upsample(outputs[:,:,0,:,:], scale_factor=8)
#         output2 = F.upsample(outputs[:,:,1,:,:], scale_factor=8)
#         output3 = F.upsample(outputs[:,:,2,:,:], scale_factor=8)
#         output = torch.stack((output1, output2, output3), 1)

        return outputs.squeeze(2)

if __name__ == '__main__':
    model = ConvLSTM()

    x = torch.randn(2, 8, 3, 256, 256)
    y = model(x)
    print(y.size())