import torch
import torch.nn as nn
from models.model import Segmentation

def build_model(args):
    device = torch.device(args.device)
    net = Segmentation(args.in_channels, args.out_channels, args.f_maps, args.n_fmaps, args.training_mode, args.fps, args.gsize, args.ext_rate)
    criterion = {
        'L1_Loss':nn.L1Loss().to(device),
        'L2_Loss':nn.MSELoss().to(device),
    }
    return net, criterion