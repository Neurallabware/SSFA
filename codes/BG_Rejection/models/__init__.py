import torch
import torch.nn as nn
from models.model import BG_Rejection

def build_model(args):
    device = torch.device(args.device)
    net = BG_Rejection(args.in_channels, args.out_channels, args.f_maps, args.gsize)

    criterion = {
        'L1_Loss': nn.L1Loss().to(device),
        'L2_Loss': nn.MSELoss().to(device),
    }

    return net, criterion