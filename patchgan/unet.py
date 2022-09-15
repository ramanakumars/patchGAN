import torch
import functools
from torch import nn
from torch.nn.parameter import Parameter
import torchvision
from collections import OrderedDict
from itertools import chain
import numpy as np

class Transferable():
    def __init__(self):
        super(Transferable, self).__init__()

    def load_transfer_data(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if param.shape == own_state[name].data.shape:
                own_state[name].copy_(param)


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

        if activation=='tanh':
            downact = nn.Tanh()
            upact   = nn.Tanh()
        else:
            downact = nn.LeakyReLU(0.2, True)
            upact   = nn.ReLU(True)

        downnorm = norm_layer(inner_nc)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = OrderedDict([(f'DownConv{layer}', downconv), 
                                (f'DownAct{layer}', downact),
                                (f'DownNorm{layer}', downnorm)])
            up = OrderedDict([(f'UpConv{layer}', upconv), 
                              (f'UpAct{layer}', nn.Sigmoid())]) ##
            if use_dropout:
                model = OrderedDict(chain(down.items(), 
                                          [(f'SubModule{layer}', submodule)],
                                          up.items()))
            else:
                model = OrderedDict(chain(down.items(),
                                          [(f'SubModule{layer}',
                                            submodule)], up.items()))#down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            #down = [downconv, downact]
            #up = [upconv, upact, upnorm]
            #model = down + up
            down = OrderedDict([(f'DownConv{layer}', downconv), 
                                (f'DownAct{layer}', downact)])
            up = OrderedDict([(f'UpConv{layer}', upconv), 
                              (f'UpAct{layer}', upact),
                              (f'UpNorm{layer}', upnorm)]) ##
            model = OrderedDict(chain(down.items(),
                                      up.items()))#down + [submodule] + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            #down = [downconv, downact, downnorm]
            #up = [upconv, upact, upnorm]
            down = OrderedDict([(f'DownConv{layer}', downconv), 
                                (f'DownAct{layer}', downact),
                                (f'DownNorm{layer}', downnorm)])
            up = OrderedDict([(f'UpConv{layer}', upconv), 
                              (f'UpAct{layer}', upact),
                              (f'UpNorm{layer}', upnorm)]) ##

            if use_dropout:
                model = OrderedDict(chain(down.items(),
                                          [(f'EncDropout{layer}', nn.Dropout(0.5))],
                                          [(f'SubModule{layer}', submodule)],
                                          up.items(),
                                          [(f'DecDropout{layer}', nn.Dropout(0.5))]))
            else:
                #model = down + [submodule] + up
                model = OrderedDict(chain(down.items(),
                                          [(f'SubModule{layer}', submodule)],
                                          up.items()))#down + [submodule] + up


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
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
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
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
    

class Discriminator(nn.Module, Transferable):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.LeakyReLU(0.2, True),
                norm_layer(ndf * nf_mult)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.LeakyReLU(0.2, True),
            norm_layer(ndf * nf_mult)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    



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
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, 
                                             activation=activation, submodule=None, 
                                             norm_layer=norm_layer, innermost=True, layer=8)  # add the innermost layer
        
        # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, activation=activation,
                                             submodule=unet_block, norm_layer=norm_layer, 
                                             use_dropout=use_dropout, layer=7)
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
