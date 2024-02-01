import torch
from torch import nn
from collections import OrderedDict
from itertools import chain


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
                               (f'DownNorm{layer}', downnorm),
                               (f'DownAct{layer}', activation),
                               ])
        if use_dropout:
            enc_sub = OrderedDict(chain(enc_sub.items(), [(f'DownDropout{layer}', nn.Dropout(0.2))]))

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
                                   (f'UpNorm{layer}', upnorm),
                                   (f'UpAct{layer}', activation),
                                   ])
        else:
            dec_sub = OrderedDict([(f'UpConv{layer}', upconv),
                                   (f'UpAct{layer}', activation)])
        if use_dropout:
            dec_sub = OrderedDict(chain(dec_sub.items(), [(f'UpDropout{layer}', nn.Dropout(0.2))]))

        self.model = nn.Sequential(dec_sub)

    def forward(self, x):
        x = self.model(x)

        return x


# custom weights initialization called on generator and discriminator
# scaling here means std
def weights_init(net, init_type='normal', scaling=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('InstanceNorm') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)


class Encoder(nn.ModuleList):
    def __init__(self, input_channels: int, gen_filts: int, gen_activation: str, use_gen_dropout: bool):
        kernel_size = 4
        padding = 1

        conv_filts = [gen_filts, gen_filts * 2, gen_filts * 4, gen_filts * 8, gen_filts * 8, gen_filts * 8, gen_filts * 8]

        encoder_layers = []

        prev_filt = input_channels
        for i, filt in enumerate(conv_filts):
            encoder_layers.append(DownSampleBlock(prev_filt, filt, gen_activation, nn.InstanceNorm2d, layer=i,
                                                  use_dropout=use_gen_dropout, kernel_size=kernel_size, stride=2,
                                                  padding=padding))
            prev_filt = filt

        super().__init__(encoder_layers)
        self.apply(weights_init)


class Decoder(nn.ModuleList):
    def __init__(self, output_channels: int, gen_filts: int, gen_activation: str, final_activation: str, use_gen_dropout: bool):
        kernel_size = 4
        padding = 1

        conv_filts = [gen_filts, gen_filts * 2, gen_filts * 4, gen_filts * 8, gen_filts * 8, gen_filts * 8, gen_filts * 8]

        prev_filt = conv_filts[-1]
        decoder_layers = []
        for i, filt in enumerate(conv_filts[:-1][::-1]):
            if i == 0:
                decoder_layers.append(UpSampleBlock(prev_filt, filt, gen_activation, nn.InstanceNorm2d, layer=i, batch_norm=False,
                                                    kernel_size=kernel_size, stride=2, padding=padding))
            else:
                decoder_layers.append(UpSampleBlock(prev_filt * 2, filt, gen_activation, nn.InstanceNorm2d, layer=i, use_dropout=use_gen_dropout,
                                                    batch_norm=True, kernel_size=kernel_size, stride=2, padding=padding))

            prev_filt = filt

        decoder_layers.append(UpSampleBlock(gen_filts * 2, output_channels, final_activation, nn.InstanceNorm2d, layer=i + 1, batch_norm=False,
                                            kernel_size=kernel_size, stride=2, padding=padding))

        super().__init__(decoder_layers)
        self.apply(weights_init)
