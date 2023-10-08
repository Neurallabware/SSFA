import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import create_feature_maps, DoubleConv, Encoder, Temporal, ResBlock, Attention


class Segmentation(nn.Module):
    """
    Neuron Segmentation.
    """

    def __init__(self, in_channels, out_channels, f_maps=32, n_fmaps=3, training_mode='recon', fps=10, gsize=5,
                 ext_rate=292.3, infer=False, layer_order='cr'):
        super().__init__()

        enlarge_size = math.ceil(gsize / 2)
        enlarge_size = enlarge_size + (enlarge_size - 1) % 2  # keep the kernel size is odd.

        self.training_mode = training_mode
        self.infer = infer

        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, n_fmaps)

        # encoder part
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, conv_kernel_size=3,
                                  apply_pooling=False, pool_kernel_size=(1, 2, 2),
                                  basic_module=DoubleConv, conv_layer_order=layer_order)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, conv_kernel_size=3,
                                  pool_kernel_size=(1, 2, 2),
                                  basic_module=DoubleConv, conv_layer_order=layer_order)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        self.res_block = [ResBlock(f_maps[-1]) for _ in range(2)]
        self.res_block = nn.Sequential(*self.res_block)

        # feature projection (ignore channels)
        self.feat_proj = nn.Sequential(
            nn.Conv3d(f_maps[-1], out_channels, kernel_size=1, stride=1, padding=0)
        )

        # attention
        self.attn = Attention(in_channels=128, enlarge_size=enlarge_size)

        # temporal modeling
        self.temporal = Temporal(fps=fps, ext_rate=ext_rate)

    def normalize(self, x):
        return torch.sigmoid(x - 5)

    def forward(self, x):
        B, _, T, H, W = x.shape
        x_scale = torch.max(x, dim=2, keepdim=True)[0]  # (B, 1, 1, H, W)

        # encoder part
        for encoder in self.encoders:
            x = encoder(x)

        # residual block
        x = self.res_block(x)  # (B, 4C, T, H/4, W/4)

        # feature projection (ignore channels)
        enc_feats = self.feat_proj(x)  # (B, 1, T, H/4, W/4)
        enc_feats = self.normalize(enc_feats)

        if self.training_mode == 'joint' or self.training_mode == 'sparse':
            enc_feats_attn_enlarge, latent, latent_enlarge = self.attn(enc_feats)

            if self.infer:
                enc_feats_attn = torch.einsum('ijkpq, ijpq->ijkpq', enc_feats, latent)
                return enc_feats, enc_feats_attn

            out = self.temporal(
                enc_feats_attn_enlarge.permute(0, 1, 3, 4, 2).reshape(-1, x.shape[2]))  # (B * 1 * H/4 * W/4, T)
        elif self.training_mode == 'recon':
            out = self.temporal(enc_feats.permute(0, 1, 3, 4, 2).reshape(-1, x.shape[2]))  # (B * 1 * H/4 * W/4, T)

        out = out.reshape(B, -1, H // 4, W // 4, T).permute(0, 1, 4, 2, 3)  # (B, 1, T, H/4, W/4)

        # decoder part
        out = F.interpolate(out, size=(T, H, W))

        # rescale
        out = (out - torch.min(out, dim=2, keepdim=True)[0]) / (
                    torch.max(out, dim=2, keepdim=True)[0] - torch.min(out, dim=2, keepdim=True)[0] + 1e-6) * x_scale

        if self.training_mode == 'joint' or self.training_mode == 'sparse':
            return out, latent, latent_enlarge, enc_feats, enc_feats_attn_enlarge
        elif self.training_mode == 'recon':
            return out, enc_feats, enc_feats, enc_feats, enc_feats
