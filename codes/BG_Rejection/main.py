import os
import argparse
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import build_dataset
from models import build_model
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=10, type=int, help='every 10 epochs drop lr')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--wo_mean', action='store_true')

    # Model parameters
    parser.add_argument('--gsize', default=8, type=int, help='neuron radius')
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--f_maps', default=32, type=int, help='initial channels of feature maps')

    # Loss coefficients
    parser.add_argument('--L1_weight', default=1.0, type=float)
    parser.add_argument('--L2_weight', default=1.0, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_name', default='', type=str)
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluate patch results in test set')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--gpu_ids', default='0, 1, 2, 3', type=str,
                        help='If GPU is used for training, specify the gpu ids in your machine')
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build dataset
    dataset_train = build_dataset(args, image_set='train')
    dataset_val = build_dataset(args, image_set='val')

    batch_num_train = len(dataset_train) // args.batch_size
    batch_num_val = len(dataset_val) // args.batch_size

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False)

    # build model
    net, criterion = build_model(args)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    # optimizer and learning strategy
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.9)

    # tensorboard for loss and output visualization
    log_dir = os.path.join(args.output_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # load pretrain model or start training from breakpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])

        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        evaluate(args, net, criterion, data_loader_val, batch_num_val, device, args.output_dir)
        return

    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_time_epoch = time.time()
        train_one_epoch(args, net, criterion, data_loader_train, batch_num_train, optimizer, device, epoch, writer)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
            # extra checkpoint every epoch
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        print('[Epoch: %d/%d] [Total time: %s] [Speed: (%.4f s/iter)]'
              % (epoch + 1,
                 args.epochs,
                 str(datetime.timedelta(seconds=int(time.time() - start_time_epoch))),
                 (time.time() - start_time_epoch) / batch_num_train))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Background Rejection Model Training and Evaluation Script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
