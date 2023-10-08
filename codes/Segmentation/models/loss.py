import torch
import torch.nn as nn


class Entropy_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_concat = torch.cat([x, 1 - x], dim=1)
        loss = -(x_concat.softmax(1) * x_concat.log_softmax(1)).sum(1).mean()
        return loss
