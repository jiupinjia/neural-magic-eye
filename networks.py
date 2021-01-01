import torch
import torch.nn as nn
from torch.nn import init
import functools
from torchvision import models

from torch.nn import ModuleList
import copy

import math
import utils
import matplotlib.pyplot as plt
import numpy as np

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PI = math.pi

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def init_weights(net, init_type='normal', init_gain=0.02):
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
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net



def define_G(args):

    norm_layer = get_norm_layer(args.norm_type)

    if args.with_disparity_conv:
        T = DisparityConv(max_shift=int(args.in_size / 4), output_nc=int(args.in_size / 4))
        G_input_nc = int(args.in_size / 4)
    else:
        T = nn.Identity()
        G_input_nc = 3

    if args.net_G == 'resnet18':
        G = ResNet(input_nc=G_input_nc, output_nc=100, m_layers=18, norm_layer=norm_layer)
    elif args.net_G == 'resnet50':
        G = ResNet(input_nc=G_input_nc, output_nc=100, m_layers=50, norm_layer=norm_layer)
    elif args.net_G == 'resnet18fcn':
        G = ResNetFCN(input_nc=G_input_nc, output_nc=1, m_layers=18, norm_layer=norm_layer,
                      with_skip_connection=args.with_skip_connection)
    elif args.net_G == 'resnet50fcn':
        G = ResNetFCN(input_nc=G_input_nc, output_nc=1, m_layers=50, norm_layer=norm_layer,
                      with_skip_connection=args.with_skip_connection)
    elif args.net_G == 'unet_64':
        G = UnetGenerator(input_nc=G_input_nc, output_nc=1, num_downs=6, ngf=64, norm_layer=norm_layer)
    elif args.net_G == 'unet_128':
        G = UnetGenerator(input_nc=G_input_nc, output_nc=1, num_downs=7, ngf=64, norm_layer=norm_layer)
    elif args.net_G == 'unet_256':
        G = UnetGenerator(input_nc=G_input_nc, output_nc=1, num_downs=8, ngf=32, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)\

    net = nn.Sequential(T, nn.ReLU(), G)

    return init_net(net)


class DisparityConv(nn.Module):

    def __init__(self, max_shift, output_nc):
        super().__init__()
        self.max_shift = int(max_shift)
        self.conv = nn.Conv2d(self.max_shift, output_nc, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):

        b, c, h, w = x.shape
        Unfold = nn.Unfold(kernel_size=(h, w), stride=(1, 1))

        # padding to the right-most coloumns
        pad_clomns = x[:, :, :, 0:self.max_shift]
        x_cat = torch.cat([x, pad_clomns], dim=-1)

        # unfold (shift along x axis and stack in channels)
        patches = Unfold(x_cat)[:, :, 1:]  # batch x embd x patches
        patches = patches.permute([0, 2, 1])  # batch x embd x patches --> # batch x patches x embd
        patches = torch.reshape(patches, [b, -1, c, h, w]) # batch x patches x c x h x w

        # compute diff maps
        x = x.unsqueeze(dim=1) # batch x 1 x c x h x w
        diff = torch.abs(x - patches)
        diff = torch.mean(diff, dim=2, keepdim=False) # batch x max_shift x h x w

        out = self.conv(diff)

        return out



class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, m_layers, norm_layer):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()

        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=True)

        if m_layers == 18:
            self.resnet = models.resnet18(pretrained=False, norm_layer=norm_layer)
            self.pred1 = nn.Linear(512, 256, bias=True)
        elif m_layers == 50:
            self.resnet = models.resnet50(pretrained=False, norm_layer=norm_layer)
            self.pred1 = nn.Linear(2048, 256, bias=True)
        else:
            raise NotImplementedError('resnet fcn m_layers [%s] is not supported' % m_layers)

        self.pred_final = nn.Linear(256, output_nc, bias=True)

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # conv in layer
        x = self.conv_in(x)

        # resnet layers
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x) # 1/4
        x = self.resnet.layer2(x) # 1/8
        x = self.resnet.layer3(x) # 1/16
        x = self.resnet.layer4(x) # 1/32

        # global avg pooling
        x = self.pooling(x) # b x c x 1 x 1
        x = torch.flatten(x, start_dim=1) # b x c

        # prediction layers
        x = self.relu(self.pred1(x))
        logits = self.pred_final(x)

        return logits


class ResNetFCN(torch.nn.Module):
    def __init__(self, input_nc, output_nc, m_layers, norm_layer,
                 with_skip_connection=False, output_sigmoid=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNetFCN, self).__init__()

        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=True)

        if m_layers == 18:
            self.resnet = models.resnet18(pretrained=False, norm_layer=norm_layer)
            self.conv_pred1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.conv_pred2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.conv_pred3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        elif m_layers == 50:
            self.resnet = models.resnet50(pretrained=False, norm_layer=norm_layer)
            self.conv_pred1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
            self.conv_pred2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.conv_pred3 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        else:
            raise NotImplementedError('resnet fcn m_layers [%s] is not supported' % m_layers)

        self.conv_pred4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_pred5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_pred_final = nn.Conv2d(64, output_nc, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.with_skip_connection = with_skip_connection

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # conv in layer
        x = self.conv_in(x)

        # resnet layers
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4
        x_8 = self.resnet.layer2(x_4) # 1/8
        x_16 = self.resnet.layer3(x_8) # 1/16
        x_32 = self.resnet.layer4(x_16) # 1/32

        # upsampling layers
        if self.with_skip_connection:
            x = self.upsamplex2(self.relu(self.conv_pred1(x_32)))  # 1/16
            x = self.upsamplex2(self.relu(self.conv_pred2(x + x_16)))  # 1/8
            x = self.upsamplex2(self.relu(self.conv_pred3(x + x_8)))  # 1/4
            x = self.upsamplex2(self.relu(self.conv_pred4(x + x_4)))  # 1/2
            x = self.upsamplex2(self.relu(self.conv_pred5(x)))  # 1/1
        else:
            x = self.upsamplex2(self.relu(self.conv_pred1(x_32))) # 1/16
            x = self.upsamplex2(self.relu(self.conv_pred2(x))) # 1/8
            x = self.upsamplex2(self.relu(self.conv_pred3(x))) # 1/4
            x = self.upsamplex2(self.relu(self.conv_pred4(x))) # 1/2
            x = self.upsamplex2(self.relu(self.conv_pred5(x))) # 1/1

        # output layers
        x = self.conv_pred_final(x) # 1/1

        if self.output_sigmoid:
            x = self.sigmoid(x)

        return x


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, output_sigmoid=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        self.conv_in = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        input_nc = ngf

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.output_sigmoid = output_sigmoid

    def forward(self, input):
        """Standard forward"""
        x = self.conv_in(input)
        x = self.model(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
