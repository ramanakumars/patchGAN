import torch
from torch import nn
from collections import OrderedDict
from itertools import chain
from .transfer import Transferable


class DownSampleBlock(nn.Module):
    def __init__(self, input_filt, output_filt, activation, norm_layer, layer, use_dropout=False, **kwargs):
        super(DownSampleBlock, self).__init__()

        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'relu':
            activation = nn.ReLU(True)
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)

        downconv = nn.Conv2d(input_filt, output_filt, **kwargs)
        downnorm = norm_layer(output_filt)

        enc_sub = OrderedDict([(f'DownConv{layer}', downconv),
                               (f'DownAct{layer}', activation),
                               (f'DownNorm{layer}', downnorm),
                               ])
        if use_dropout:
            enc_sub = OrderedDict(chain(enc_sub.items(),
                                        [(f'DownDropout{layer}', nn.Dropout(0.2))]))

        self.model = nn.Sequential(enc_sub)

    def forward(self, x):
        x = self.model(x)

        return x


class UpSampleBlock(nn.Module):
    def __init__(self, input_filt, output_filt, activation, norm_layer, layer, batch_norm=True, use_dropout=False, **kwargs):
        super(UpSampleBlock, self).__init__()

        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'relu':
            activation = nn.ReLU(True)
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)
        elif activation == 'softmax':
            activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()

        upconv = nn.ConvTranspose2d(input_filt, output_filt, **kwargs)
        if batch_norm:
            upnorm = norm_layer(output_filt)
            dec_sub = OrderedDict([(f'UpConv{layer}', upconv),
                                   (f'UpAct{layer}', activation),
                                   (f'UpNorm{layer}', upnorm)])
        else:
            dec_sub = OrderedDict([(f'UpConv{layer}', upconv),
                                   (f'UpAct{layer}', activation)])
        if use_dropout:
            dec_sub = OrderedDict(chain(dec_sub.items(),
                                        [(f'UpDropout{layer}', nn.Dropout(0.2))]))

        self.model = nn.Sequential(dec_sub)

    def forward(self, x):
        x = self.model(x)

        return x


class UNet(nn.Module, Transferable):
    def __init__(self, input_nc, output_nc, nf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 activation='tanh', final_act='softmax'):
        super(UNet, self).__init__()

        kernel_size = 4
        padding = 1

        conv_filts = [nf, nf * 2, nf * 4, nf * 8, nf * 8, nf * 8, nf * 8]

        encoder_layers = []

        prev_filt = input_nc
        for i, filt in enumerate(conv_filts):
            encoder_layers.append(DownSampleBlock(prev_filt, filt, activation, norm_layer, layer=i,
                                                  use_dropout=use_dropout, kernel_size=kernel_size, stride=2,
                                                  padding=padding, bias=False))
            prev_filt = filt

        decoder_layers = []
        for i, filt in enumerate(conv_filts[:-1][::-1]):
            if i == 0:
                decoder_layers.append(UpSampleBlock(prev_filt, filt, activation, norm_layer, layer=i, batch_norm=False,
                                                    kernel_size=kernel_size, stride=2, padding=padding, bias=False))
            else:
                decoder_layers.append(UpSampleBlock(prev_filt * 2, filt, activation, norm_layer, layer=i, use_dropout=use_dropout,
                                                    batch_norm=True, kernel_size=kernel_size, stride=2, padding=padding, bias=False))

            prev_filt = filt

        decoder_layers.append(UpSampleBlock(nf * 2, output_nc, final_act, norm_layer, layer=i + 1, batch_norm=False,
                                            kernel_size=kernel_size, stride=2, padding=padding, bias=False))

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x, return_hidden=False):
        xencs = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            xencs.append(x)

        hidden = xencs[-1]

        xencs = xencs[::-1]

        for i, layer in enumerate(self.decoder):
            if i == 0:
                xinp = hidden
            else:
                xinp = torch.cat([x, xencs[i]], dim=1)

            x = layer(xinp)

        if return_hidden:
            return x, hidden
        else:
            return x
