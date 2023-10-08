import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrLoss(nn.Module):
    # correlation loss
    def __init__(self, patch_size):
        super().__init__()

        self.kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.mask = torch.ones((1, 1, patch_size[1], patch_size[2]), dtype=torch.float32)
        self.mask = F.conv2d(self.mask, self.kernel, stride=1, padding=1)

        self.patch_size = patch_size
        self.eps = 1e-6  # avoid NaN

    def forward(self, x, mask):
        # preprocess
        x = x - torch.mean(x, dim=2, keepdim=True)
        x1 = torch.sqrt(torch.mean(x * x, dim=2, keepdim=True))
        x = x / (x1 + self.eps)

        x_conv = F.conv2d(x.permute(0, 2, 1, 3, 4).flatten(0, 1), self.kernel.to(x.device), stride=1,
                          padding=1).reshape(-1, self.patch_size[0], 1, self.patch_size[1], self.patch_size[2]).permute(
            0, 2, 1, 3, 4)

        out = torch.mean(x * x_conv, dim=2) / self.mask.to(x.device)

        return torch.sum(out * mask.squeeze(1)) / (mask.sum() + self.eps)


class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        B, C, T, H, W = x.shape
        tv_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).sum()
        return self.weight * (tv_h + tv_w) / (B * C * T * H * W)
