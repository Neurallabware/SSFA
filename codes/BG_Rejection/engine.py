import os
import time
import datetime
import tifffile
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable
from statistics import mean


def train_one_epoch(args, net: torch.nn.Module, criterion: dict,
                    data_loader: Iterable, batch_num_train: int,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, writer: SummaryWriter):
    net.train()

    vis_dir = os.path.join(args.output_dir, 'vis_train')
    os.makedirs(vis_dir, exist_ok=True)

    print_freq = 10

    L1_Loss_train = []
    L2_Loss_train = []
    start_time = time.time()
    for iteration, input in enumerate(data_loader, 1):
        input = input.to(device)

        # forward
        out_bg, out_neuron = net(input)

        # calculate loss
        L1_Loss = criterion['L1_Loss'](out_neuron, (input - out_bg)[:, :, 1::2])
        L2_Loss = criterion['L2_Loss'](out_neuron, (input - out_bg)[:, :, 1::2])

        Total_Loss = args.L1_weight * L1_Loss + args.L2_weight * L2_Loss

        # backward
        optimizer.zero_grad()
        Total_Loss.backward()
        optimizer.step()

        # accumulate average loss in a batch
        L1_Loss_train.append(L1_Loss.item())
        L2_Loss_train.append(L2_Loss.item())

        left_time = datetime.timedelta(
            seconds=int(((time.time() - start_time) / iteration) * (batch_num_train - iteration)))
        if iteration % print_freq == 0:
            print('[Epoch: %d/%d] [Batch: %d/%d] [ETA: %s] [L1_Loss: %.4f L2_Loss: %.4f]'
                  % (epoch + 1,
                     args.epochs,
                     iteration,
                     batch_num_train,
                     str(left_time),
                     mean(L1_Loss_train),
                     mean(L2_Loss_train)))

        if (epoch * batch_num_train + iteration) % 100 == 0:
            writer.add_scalar('L1_Loss_iter', mean(L1_Loss_train), (epoch * batch_num_train + iteration))
            writer.add_scalar('L2_Loss_iter', mean(L2_Loss_train), (epoch * batch_num_train + iteration))

    # Loss visualization
    writer.add_scalar('L1_Loss', mean(L1_Loss_train), epoch + 1)
    writer.add_scalar('L2_Loss', mean(L2_Loss_train), epoch + 1)

    # Video output visualization
    if (epoch + 1) % 1 == 0:
        if args.wo_mean:
            input_vis = (input.detach()[-1].squeeze().cpu().numpy() + np.load(
                'mean_train_%s.npy' % args.dataset_name)).clip(0, 65535)[::2, :, :]
            out_bg_vis = (out_bg.detach()[-1].squeeze().cpu().numpy() + np.load(
                'mean_train_%s.npy' % args.dataset_name)).clip(0, 65535)[::2, :, :]
            out_neuron_vis = out_neuron.detach()[-1].squeeze().cpu().numpy().clip(0, 65535)
        else:
            input_vis = input.detach()[-1].squeeze().cpu().numpy().clip(0, 65535)[::2, :, :]
            out_bg_vis = out_bg.detach()[-1].squeeze().cpu().numpy().clip(0, 65535)[::2, :, :]
            out_neuron_vis = out_neuron.detach()[-1].squeeze().cpu().numpy().clip(0, 65535)

        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_input.tiff' % (epoch + 1)), input_vis.astype(np.uint16))
        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_out_bg.tiff' % (epoch + 1)), out_bg_vis.astype(np.uint16))
        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_out_neuron.tiff' % (epoch + 1)),
                         out_neuron_vis.astype(np.uint16))


@torch.no_grad()
def evaluate(args, net, criterion, data_loader, batch_num_val, device, output_dir):
    net.eval()

    save_dir = os.path.join(output_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    print_freq = 10

    L1_Loss_val = []
    L2_Loss_val = []
    start_time = time.time()
    for iteration, input in enumerate(data_loader, 1):
        bs = input.shape[0]
        input = input.to(device)

        # forward
        out_bg, out_neuron = net(input)

        L1_Loss = criterion['L1_Loss'](out_neuron, (input - out_bg)[:, :, 1::2])
        L2_Loss = criterion['L2_Loss'](out_neuron, (input - out_bg)[:, :, 1::2])
        L1_Loss_val.append(L1_Loss.item())
        L2_Loss_val.append(L2_Loss.item())

        if args.wo_mean:
            input_vis = (input.squeeze(1).cpu().numpy() + np.load('mean_val_%s.npy' % args.dataset_name)).clip(0,
                                                                                                               65535)[
                        ::2, :, :]
            out_bg_vis = (out_bg.squeeze(1).cpu().numpy() + np.load('mean_val_%s.npy' % args.dataset_name)).clip(0,
                                                                                                                 65535)[
                         ::2, :, :]
            out_neuron_vis = out_neuron.squeeze(1).cpu().numpy().clip(0, 65535)
        else:
            input_vis = input.squeeze(1).cpu().numpy().clip(0, 65535)[::2, :, :]
            out_bg_vis = out_bg.squeeze(1).cpu().numpy().clip(0, 65535)[::2, :, :]
            out_neuron_vis = out_neuron.squeeze(1).cpu().numpy().clip(0, 65535)

        for i in range(bs):
            count = bs * (iteration - 1) + i + 1
            tifffile.imwrite(os.path.join(save_dir, 'input_%d.tiff') % count, input_vis[i].astype(np.uint16))
            tifffile.imwrite(os.path.join(save_dir, 'out_bg_%d.tiff') % count, out_bg_vis[i].astype(np.uint16))
            tifffile.imwrite(os.path.join(save_dir, 'out_neuron_%d.tiff') % count, out_neuron_vis[i].astype(np.uint16))

        left_time = datetime.timedelta(
            seconds=int(((time.time() - start_time) / iteration) * (batch_num_val - iteration)))
        if iteration % print_freq == 0:
            print('[Evaluation] [Batch: %d/%d] [ETA: %s] [L1_Loss: %.4f L2_Loss: %.4f]'
                  % (iteration,
                     batch_num_val,
                     str(left_time),
                     mean(L1_Loss_val),
                     mean(L2_Loss_val)))
