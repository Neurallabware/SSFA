import os
import time
import datetime
import tifffile
import argparse
import numpy as np
import torch
import utils
from models.model import BG_Rejection


def get_args_parser():
    parser = argparse.ArgumentParser('Full Image inference', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--wo_mean', action='store_true')

    parser.add_argument('--patch_size', default='', type=str)
    parser.add_argument('--stride_size', default='', type=str)

    parser.add_argument('--gsize', default=8, type=int, help='neuron radius')
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--f_maps', default=32, type=int, help='initial channels of feature maps')

    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--ckpt_dir', default='', type=str)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_ids', default='0, 1, 2, 3', type=str,
                        help='If GPU is used for inference, specify the gpu ids in your machine')
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device(args.device)

    patch_size = eval(args.patch_size)
    stride_size = eval(args.stride_size)
    overlap_size = (patch_size[0] - stride_size[0],
                    patch_size[1] - stride_size[1],
                    patch_size[2] - stride_size[2])

    # load model and pretrained weights
    net = BG_Rejection(in_channels=args.in_channels, out_channels=args.out_channels, f_maps=args.f_maps,
                       gsize=args.gsize, infer=True)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)

    checkpoint = torch.load(os.path.join(args.ckpt_dir, 'checkpoint.pth'), map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    # load data
    video_list = os.listdir(args.dataset_dir)
    video_list.sort(key=lambda x: int(x.split('.')[0]))

    with torch.no_grad():
        net.eval()

        start_time = time.time()
        for i in range(len(video_list)):
            video = tifffile.imread(os.path.join(args.dataset_dir, video_list[i])).astype(np.float32)
            origin_shape = video.shape
            if args.wo_mean:
                video_mean = np.mean(video, axis=0, keepdims=True)
                video = video - video_mean
            video_patches, patch_num = utils.crop_patches(video, patch_size, stride_size)

            iter_num = len(video_patches) // args.batch_size if len(video_patches) % args.batch_size == 0 else len(
                video_patches) // args.batch_size + 1

            bg_patches = []
            neuron_patches = []
            for j in range(iter_num):
                input = torch.as_tensor(video_patches[j * args.batch_size:(j + 1) * args.batch_size],
                                        dtype=torch.float32, device=device).unsqueeze(1)
                out_bg, out_neuron = net(input)

                out_bg = out_bg.squeeze(1)
                out_neuron = out_neuron.squeeze(1)

                bg_patches.append(out_bg.cpu())
                neuron_patches.append(out_neuron.cpu())

                left_time = datetime.timedelta(seconds=int((time.time() - start_time) / (i * iter_num + j + 1) * (
                            (len(video_list) - i - 1) * iter_num + (iter_num - j - 1))))
                print('[Inference Video: %d/%d] [Pacthes: %d/%d] [ETA: %s]' % (
                i + 1, len(video_list), j + 1, iter_num, str(left_time)))

            bg_patches = torch.cat(bg_patches, dim=0).numpy()
            neuron_patches = torch.cat(neuron_patches, dim=0).numpy()

            out_bg = utils.concat_patches(bg_patches, origin_shape, patch_num, patch_size, stride_size, overlap_size)
            out_neuron = utils.concat_patches(neuron_patches, origin_shape, patch_num, patch_size, stride_size,
                                              overlap_size)

            if args.wo_mean:
                out_bg = (out_bg + video_mean).clip(0, 65535)
                out_neuron = out_neuron.clip(0, 65535)
            else:
                out_bg = out_bg.clip(0, 65535)
                out_neuron = out_neuron.clip(0, 65535)

            tifffile.imwrite(os.path.join(args.output_dir, 'bg', video_list[i]), out_bg.astype(np.uint16))
            tifffile.imwrite(os.path.join(args.output_dir, 'rmbg_wo_postprocessing', video_list[i]),
                             out_neuron.astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SSFA inference scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, 'bg'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'rmbg_wo_postprocessing'), exist_ok=True)
    main(args)
