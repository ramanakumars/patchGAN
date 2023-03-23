import torch
import functools
from torch import nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
from itertools import chain


class Transferable():
    def __init__(self):
        super(Transferable, self).__init__()

    def load_transfer_data(self, checkpoint, verbose=False):
        state_dict = torch.load(checkpoint, map_location=next(self.parameters()).device)
        own_state = self.state_dict()
        state_names = list(own_state.keys())
        count = 0
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            # find the weight with the closest name to this
            sub_name = '.'.join(name.split('.')[-2:])
            own_state_name = [n for n in state_names if sub_name in n]
            if len(own_state_name) == 1:
                own_state_name = own_state_name[0]
            else:
                if verbose:
                    print(f'{name} not found')
                continue

            if param.shape == own_state[own_state_name].data.shape:
                own_state[own_state_name].copy_(param)
                count += 1

        if count == 0:
            print("WARNING: Could not transfer over any weights!")
        else:
            print(f"Loaded weights for {count} layers")


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 activation='tanh', norm_layer=nn.BatchNorm2d, use_dropout=False, layer=1):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)

        if activation == 'tanh':
            downact = nn.Tanh()
            upact = nn.Tanh()
        else:
            downact = nn.LeakyReLU(0.2, True)
            upact = nn.ReLU(True)

        downnorm = norm_layer(inner_nc)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

            if outer_nc == 1:
                upact = nn.Sigmoid()
            else:
                upact = nn.Softmax(dim=1)
            down = OrderedDict([(f'DownConv{layer}', downconv),
                                (f'DownAct{layer}', downact),
                                (f'DownNorm{layer}', downnorm)])
            up = OrderedDict([(f'UpConv{layer}', upconv),
                              (f'UpAct{layer}', upact)])
            if use_dropout:
                model = OrderedDict(chain(down.items(),
                                          [(f'SubModule{layer}', submodule)],
                                          up.items()))
            else:
                model = OrderedDict(chain(down.items(),
                                          [(f'SubModule{layer}',
                                            submodule)], up.items()))  # down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = OrderedDict([(f'DownConv{layer}', downconv),
                                (f'DownAct{layer}', downact)])
            up = OrderedDict([(f'UpConv{layer}', upconv),
                              (f'UpAct{layer}', upact),
                              (f'UpNorm{layer}', upnorm)])
            model = OrderedDict(chain(down.items(),
                                      up.items()))  # down + [submodule] + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = OrderedDict([(f'DownConv{layer}', downconv),
                                (f'DownAct{layer}', downact),
                                (f'DownNorm{layer}', downnorm)])
            up = OrderedDict([(f'UpConv{layer}', upconv),
                              (f'UpAct{layer}', upact),
                              (f'UpNorm{layer}', upnorm)])

            if use_dropout:
                model = OrderedDict(chain(down.items(),
                                          [(f'EncDropout{layer}',
                                            nn.Dropout(0.5))],
                                          [(f'SubModule{layer}', submodule)],
                                          up.items(),
                                          [(f'DecDropout{layer}', nn.Dropout(0.5))]))
            else:
                model = OrderedDict(chain(down.items(),
                                          [(f'SubModule{layer}', submodule)],
                                          up.items()))  # down + [submodule] + up

        self.model = nn.Sequential(model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


def get_norm_layer():
    """Return a normalization layer
       For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    """
    norm_type = 'batch'
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer


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
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class UnetGenerator(nn.Module, Transferable):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, nf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 activation='tanh'):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            nf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, activation=activation,
                                             submodule=None, norm_layer=norm_layer,
                                             use_dropout=use_dropout, innermost=True, layer=7)

        # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, activation=activation,
                                             submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout, layer=6)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, activation=activation,
                                             submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout, layer=5)

        # gradually reduce the number of filters from nf * 8 to nf
        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, activation=activation,
                                             submodule=unet_block, norm_layer=norm_layer, layer=4)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, activation=activation,
                                             submodule=unet_block, norm_layer=norm_layer, layer=3)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, activation=activation,
                                             submodule=unet_block, norm_layer=norm_layer, layer=2)
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, activation=activation,
                                             submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, layer=1)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


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
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
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
