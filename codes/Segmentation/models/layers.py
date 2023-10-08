import torch
import torch.nn as nn
import torch.nn.functional as F


def create_feature_maps(f_maps, n_fmaps):
    return [f_maps * 2 ** n for n in range(n_fmaps)]


def create_conv(in_channels, out_channels, kernel_size, order, num_groups=8):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Parameters:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): 3d conv kernel size
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            use_bias = not ('g' in order or 'b' in order)
            padding = kernel_size // 2
            modules.append(
                ('Conv3d', nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=use_bias)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('GroupNorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('BatchNorm3d', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('BatchNorm3d', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")
    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cr'):
        """
        Parameters:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): 3d conv kernel size
            order (string): determines the order of layers, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
                'cl' -> conv + LeakyReLU
                'ce' -> conv + ELU
        """
        super().__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cr'):
        """
        Parameters:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
            kernel_size (int): 3d conv kernel size
            order (string): determines the order of layers, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
                'cl' -> conv + LeakyReLU
                'ce' -> conv + ELU
        """
        super().__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order))


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='cr'):
        """
        Parameters:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            conv_kernel_size (int): 3d conv kernel size
            apply_pooling (bool): if True use MaxPool3d before DoubleConv
            pool_kernel_size (tuple): the size of the window to take a max over
            pool_type (str): pooling layer: 'max' or 'avg'
            basic_module(nn.Module): either ResNetBlock or DoubleConv
            conv_layer_order (string): determines the order of layers in `DoubleConv` module. See `DoubleConv` for more info.
        """
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv module.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr'):
        super().__init__()
        """
        Parameters:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): 3d conv kernel size
            scale_factor (tuple): used as the multiplier for the image H/W/D in
                case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
                from the corresponding encoder
            basic_module(nn.Module): either ResNetBlock or DoubleConv
            conv_layer_order (string): determines the order of layers
                in `DoubleConv` module. See `DoubleConv` for more info.
        """
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order)

    def forward(self, x):
        if self.upsample is None:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
            # x += encoder_features

        x = self.basic_module(x)
        return x


class Temporal(nn.Module):
    def __init__(self, fps=10, ext_rate=292.3):
        super().__init__()

        self.cal_params = {}
        self.cal_params['a_bind'] = 3.5
        self.cal_params['a_ubind'] = 7
        self.scale = nn.parameter.Parameter(torch.tensor([1e5], dtype=torch.float32), requires_grad=False)

        # parameters
        self.cal_params['dt'] = 1 / fps
        self.cal_params['ext_rate'] = ext_rate
        self.cal_params['ca_bind'] = 110
        self.cal_params['t_on'] = 0.8535
        self.cal_params['t_off'] = 98.6173
        self.cal_params['ca_amp'] = 76.1251
        self.cal_params['ca_rest'] = 50e-9
        self.cal_params['ca_dis'] = 290e-9
        self.cal_params['ind_con'] = 200e-6
        self.cal_params['ca_sat'] = 1

        # ToTensor
        for key in self.cal_params:
            self.cal_params[key] = torch.as_tensor(self.cal_params[key], dtype=torch.float32)

    def forward(self, spikes):
        C = torch.zeros_like(spikes, dtype=torch.float32, device=spikes.device)
        C[:, 0] = self.cal_params['ca_rest'] * self.scale

        for k in range(1, spikes.shape[1]):
            C[:, k] = C[:, k - 1] + (-self.cal_params['dt'] * self.cal_params['ext_rate'] * (C[:, k - 1] \
                                                                                             - self.cal_params[
                                                                                                 'ca_rest'] * self.scale) + spikes[
                                                                                                                            :,
                                                                                                                            k]) / (
                                  1 + self.cal_params['ca_bind'] + \
                                  (self.cal_params['ind_con'] * self.scale * self.cal_params['ca_dis'] * self.scale) / (
                                              C[:, k - 1] + self.cal_params['ca_dis'] * self.scale) ** 2)
            if self.cal_params['ca_sat'] < 1 and self.cal_params['ca_sat'] >= 0:
                C[:, k] = torch.where(C[:, k] <= self.cal_params['ca_dis'] * self.scale * self.cal_params['ca_sat'] / (
                            1 - self.cal_params['ca_sat']), C[:, k],
                                      self.cal_params['ca_dis'] * self.scale * self.cal_params['ca_sat'] / (
                                                  1 - self.cal_params['ca_sat']))

        F = 25.2 * (1 / (1 + (290e-9 * self.scale / C) ** 2.7))
        F = F - torch.min(F, 1, keepdim=True)[0]
        return F


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)


class Attention(nn.Module):
    def __init__(self, in_channels=128, enlarge_size=3):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.temporal_squeeze = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        )
        self.enlarge = nn.MaxPool2d(kernel_size=enlarge_size, stride=1, padding=enlarge_size // 2)

    def forward(self, enc_feats):
        """
        enc_feats: encoded feature maps followed by a feature projection, shape=(B, 1, T, H/4, W/4)
        """
        # latent projection and temporal squeeze(ignore temporal signal and only save footprint)
        latent = self.spatial_conv(enc_feats)  # (B, 1, T, H/4, W/4)
        latent = self.temporal_squeeze(latent.squeeze(1))  # (B, 1, H/4, W/4)

        latent = torch.sigmoid(10 * latent)  # default setting

        # enlarge
        latent_enlarge = self.enlarge(latent)
        latent_enlarge[latent_enlarge < 0.1] = 0

        # attention
        enc_feats_attn_enlarge = torch.einsum('ijkpq, ijpq->ijkpq', enc_feats, latent_enlarge)

        return enc_feats_attn_enlarge, latent, latent_enlarge
