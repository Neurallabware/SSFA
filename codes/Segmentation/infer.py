import os
import time
import datetime
import tifffile
import argparse
import numpy as np
import torch
import utils
from models.model import Segmentation


def get_args_parser():
    parser = argparse.ArgumentParser('Full Image inference', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--wo_min', action='store_true')
    parser.add_argument('--normalize_factor', default=1, type=int)

    parser.add_argument('--patch_size', default='(128, 256, 256)', type=str)
    parser.add_argument('--stride_size', default='(96, 224, 224)', type=str)
    parser.add_argument('--latent_patch_size', default='(128, 64, 64)', type=str)
    parser.add_argument('--latent_stride_size', default='(96, 56, 56)', type=str)

    parser.add_argument('--training_mode', default='joint', type=str)
    parser.add_argument('--fps', default=15, type=int, help='frame rate')
    parser.add_argument('--gsize', default=5, type=int, help='neuron size')
    parser.add_argument('--ext_rate', default=292.3, type=float, help='temporal extrusion rate')
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--f_maps', default=32, type=int, help='initial channels of feature maps')
    parser.add_argument('--n_fmaps', default=3, type=int, help='feature map levels')

    parser.add_argument('--dataset_name', default='', type=str)
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--ckpt_dir', default='', type=str)
    parser.add_argument('--ckpt_dir2', default='', type=str)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_ids', default='', type=str,
                        help='If GPU is used for inference, specify the gpu ids in your machine')
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device(args.device)

    patch_size = eval(args.patch_size)
    stride_size = eval(args.stride_size)

    latent_patch_size = eval(args.latent_patch_size)
    latent_stride_size = eval(args.latent_stride_size)

    latent_overlap_size = (latent_patch_size[0] - latent_stride_size[0],
                           latent_patch_size[1] - latent_stride_size[1],
                           latent_patch_size[2] - latent_stride_size[2])

    # load model and pretrained weights
    net = Segmentation(args.in_channels, args.out_channels, args.f_maps, args.n_fmaps, args.training_mode, args.fps,
                       args.gsize, args.ext_rate, infer=True)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)
    if args.training_mode == 'joint':
        checkpoint = torch.load(os.path.join(args.ckpt_dir, 'checkpoint.pth'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
    elif args.training_mode == 'sparse':
        checkpoint_recon = torch.load(os.path.join(args.ckpt_dir, 'checkpoint.pth'), map_location='cpu')
        net.load_state_dict(checkpoint_recon['net'], strict=False)
        checkpoint_sparse = torch.load(os.path.join(args.ckpt_dir2, 'checkpoint.pth'), map_location='cpu')
        net.load_state_dict(checkpoint_sparse['net'], strict=False)

    # load data
    video_list = os.listdir(args.dataset_dir)
    video_list.sort(key=lambda x: int(x.split('.')[0]))

    with torch.no_grad():
        net.eval()

        start_time = time.time()

        for i in range(len(video_list)):
            video = tifffile.imread(os.path.join(args.dataset_dir, video_list[i])).astype(np.float32)
            latent_shape = (video.shape[0], video.shape[1] // 4, video.shape[2] // 4)
            if args.wo_min:
                video_min = np.min(video, axis=0, keepdims=True)
                video_input = (video - video_min) / args.normalize_factor
            else:
                video_input = video / args.normalize_factor
            video_patches, patch_num = utils.crop_patches(video_input, patch_size, stride_size)

            iter_num = len(video_patches) // args.batch_size if len(video_patches) % args.batch_size == 0 else len(
                video_patches) // args.batch_size + 1

            enc_feats_patches = []
            enc_feats2_patches = []
            for j in range(iter_num):
                input = torch.as_tensor(video_patches[j * args.batch_size:(j + 1) * args.batch_size],
                                        dtype=torch.float32, device=device).unsqueeze(1)
                enc_feats, enc_feats2 = net(input)

                enc_feats_patches.append(enc_feats.squeeze(1).cpu())
                enc_feats2_patches.append(enc_feats2.squeeze(1).cpu())

                left_time = datetime.timedelta(seconds=int((time.time() - start_time) / (i * iter_num + j + 1) * (
                            (len(video_list) - i - 1) * iter_num + (iter_num - j - 1))))
                print('[Inference Video: %d/%d] [Pacthes: %d/%d] [ETA: %s]' % (
                i + 1, len(video_list), j + 1, iter_num, str(left_time)))

            enc_feats_patches = torch.cat(enc_feats_patches, dim=0).cpu().numpy()
            enc_feats2_patches = torch.cat(enc_feats2_patches, dim=0).cpu().numpy()

            enc_feats = utils.concat_patches(enc_feats_patches, latent_shape, patch_num, latent_patch_size,
                                             latent_stride_size, latent_overlap_size)
            enc_feats2 = utils.concat_patches(enc_feats2_patches, latent_shape, patch_num, latent_patch_size,
                                              latent_stride_size, latent_overlap_size)

            tifffile.imwrite(os.path.join(args.output_dir, 'enc_feats' + args.dataset_name, video_list[i]),
                             (enc_feats * 65535).astype(np.uint16))
            tifffile.imwrite(os.path.join(args.output_dir, 'latent' + args.dataset_name, video_list[i]),
                             (enc_feats2 * 65535).astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Segmentation Model Inference Scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, 'enc_feats' + args.dataset_name), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'latent' + args.dataset_name), exist_ok=True)
    main(args)
