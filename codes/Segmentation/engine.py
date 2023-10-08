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
                    data_loader: Iterable, batch_num_train:int,
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, writer: SummaryWriter):
    net.train()

    vis_dir = os.path.join(args.output_dir, 'vis_train')
    os.makedirs(vis_dir, exist_ok=True)

    print_freq = 10

    L1_Loss_train = []
    L2_Loss_train = []
    Sparse_Loss_train = []
    start_time = time.time()
    for iteration, input in enumerate(data_loader, 1):
        input = input.to(device)

        # forward
        output, latent, latent_enlarge, enc_feats, enc_feats_attn_enlarge = net(input)

        # save epoch 0
        # input_vis = (input.detach()[-2].squeeze().cpu().numpy() * args.normalize_factor + np.load('min_train_%s.npy' % args.dataset_name)).clip(0, 65535).astype(np.uint16)
        # output_vis = (output.detach()[-2].squeeze().cpu().numpy() * args.normalize_factor + np.load('min_train_%s.npy' % args.dataset_name)).clip(0, 65535).astype(np.uint16)
        # latent_vis = (latent.detach()[-2].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
        # latent_enlarge_vis = (latent_enlarge.detach()[-2].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
        # enc_feats_vis = (enc_feats.detach()[-2].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
        # enc_feats_attn_enlarge_vis = (enc_feats_attn_enlarge.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)

        # tifffile.imwrite(os.path.join(vis_dir, 'Epoch0_input.tiff'), input_vis)
        # tifffile.imwrite(os.path.join(vis_dir, 'Epoch0_output.tiff'), output_vis)
        # tifffile.imwrite(os.path.join(vis_dir, 'Epoch0_latent.tiff'), latent_vis)
        # tifffile.imwrite(os.path.join(vis_dir, 'Epoch0_latent_enlarge.tiff'), latent_enlarge_vis)
        # tifffile.imwrite(os.path.join(vis_dir, 'Epoch0_enc_feats.tiff'), enc_feats_vis)
        # tifffile.imwrite(os.path.join(vis_dir, 'Epoch0_enc_feats_attn_enlarge.tiff'), enc_feats_attn_enlarge_vis)

        # calculate loss
        L1_Loss = criterion['L1_Loss'](output, input)
        L2_Loss = criterion['L2_Loss'](output, input)
        Sparse_Loss = criterion['L1_Loss'](latent, torch.zeros_like(latent))

        Total_Loss = args.L1_weight * L1_Loss + args.L2_weight * L2_Loss + args.Sparse_weight * Sparse_Loss
        
        # backward
        optimizer.zero_grad()
        Total_Loss.backward()
        optimizer.step()

        # accumulate average loss in a batch
        L1_Loss_train.append(L1_Loss.item())
        L2_Loss_train.append(L2_Loss.item())
        Sparse_Loss_train.append(Sparse_Loss.item())
        
        left_time = datetime.timedelta(seconds= int(((time.time() - start_time) / iteration) * (batch_num_train - iteration)))
        if iteration % print_freq == 0:
            print('[Epoch: %d/%d] [Batch: %d/%d] [ETA: %s] [L1_Loss: %.4f L2_Loss: %.4f Sparse_Loss: %.6f]'
            %(epoch + 1,
                args.epochs,
                iteration,
                batch_num_train,
                str(left_time),
                mean(L1_Loss_train),
                mean(L2_Loss_train),
                mean(Sparse_Loss_train)))
        
        if (epoch * batch_num_train + iteration) % 10 == 0:
            writer.add_scalar('L1_Loss_iter', mean(L1_Loss_train), (epoch * batch_num_train + iteration))
            writer.add_scalar('L2_Loss_iter', mean(L2_Loss_train), (epoch * batch_num_train + iteration))
            writer.add_scalar('Sparse_Loss_iter', mean(Sparse_Loss_train), (epoch * batch_num_train + iteration))

    # Loss visualization
    writer.add_scalar('L1_Loss', mean(L1_Loss_train), epoch + 1)
    writer.add_scalar('L2_Loss', mean(L2_Loss_train), epoch + 1)
    writer.add_scalar('Sparse_Loss', mean(Sparse_Loss_train), epoch + 1)

    # Video output visualization
    if (epoch + 1) % 1 == 0:
        if args.wo_min:
            input_vis = (input.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor + np.load('min_train_%s.npy' % args.dataset_name)).clip(0, 65535).astype(np.uint16)
            output_vis = (output.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor + np.load('min_train_%s.npy' % args.dataset_name)).clip(0, 65535).astype(np.uint16)
            latent_vis = (latent.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
            latent_enlarge_vis = (latent_enlarge.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
            enc_feats_vis = (enc_feats.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
            enc_feats_attn_enlarge_vis = (enc_feats_attn_enlarge.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
        else:
            input_vis = (input.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor).clip(0, 65535).astype(np.uint16)
            output_vis = (output.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor).clip(0, 65535).astype(np.uint16)
            latent_vis = (latent.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
            latent_enlarge_vis = (latent_enlarge.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
            enc_feats_vis = (enc_feats.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
            enc_feats_attn_enlarge_vis = (enc_feats_attn_enlarge.detach()[-1].squeeze().cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)

        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_input.tiff' %(epoch + 1)), input_vis)
        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_output.tiff' %(epoch + 1)), output_vis)
        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_latent.tiff' %(epoch + 1)), latent_vis)
        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_latent_enlarge.tiff' %(epoch + 1)), latent_enlarge_vis)
        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_enc_feats.tiff' %(epoch + 1)), enc_feats_vis)
        tifffile.imwrite(os.path.join(vis_dir, 'Epoch%d_enc_feats_attn_enlarge.tiff' %(epoch + 1)), enc_feats_attn_enlarge_vis)

@torch.no_grad()
def evaluate(args, net, criterion, data_loader, batch_num_val, device, output_dir):
    net.eval()

    save_dir = os.path.join(output_dir, 'eval')
    os.makedirs(save_dir, exist_ok=True)

    print_freq = 10

    L1_Loss_val = []
    L2_Loss_val = []
    start_time = time.time()
    for iteration, input in enumerate(data_loader, 1):
        input = input.to(device)

        # forward
        output, latent_feats = net(input)

        L1_Loss = criterion['L1_Loss'](output, input)
        L2_Loss = criterion['L2_Loss'](output, input)
        L1_Loss_val.append(L1_Loss.item())
        L2_Loss_val.append(L2_Loss.item())

        if args.wo_min:
            input_vis = (input.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor + np.load('min_val_%s.npy' % args.dataset_name)).clip(0, 65535).astype(np.uint16)
            output_vis = (output.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor + np.load('min_val_%s.npy' % args.dataset_name)).clip(0, 65535).astype(np.uint16)
            seg_vis = (latent_feats.detach()[-1].squeeze().cpu().numpy() * 65535).astype(np.uint16)
        else:
            input_vis = (input.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor).clip(0, 65535).astype(np.uint16)
            output_vis = (output.detach()[-1].squeeze().cpu().numpy() * args.normalize_factor).clip(0, 65535).astype(np.uint16)
            seg_vis = (latent_feats.detach()[-1].squeeze().cpu().numpy() * 65535).astype(np.uint16)
        tifffile.imwrite(os.path.join(save_dir, 'input_%d.tiff') % iteration, input_vis)
        tifffile.imwrite(os.path.join(save_dir, 'output_%d.tiff') % iteration, output_vis)
        tifffile.imwrite(os.path.join(save_dir, 'seg_%d.tiff') % iteration, seg_vis)

        left_time = datetime.timedelta(seconds= int(((time.time() - start_time) / iteration) * (batch_num_val - iteration)))
        if iteration % print_freq == 0:
            print('[Evaluation] [Batch: %d/%d] [ETA: %s] [L1_Loss: %.4f L2_Loss: %.4f]'
            %(iteration,
              batch_num_val,
              str(left_time),
              mean(L1_Loss_val),
              mean(L2_Loss_val)))
