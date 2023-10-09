import os
import tifffile
import numpy as np
import argparse
import random


def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)
    # Patches
    parser.add_argument('--patch_size', default='(128, 128, 128)', type=str, help='(T, H, W)')
    parser.add_argument('--n_patch_all', default=150, type=int, help='random patch number of each whole video')
    parser.add_argument('--n_patch_train', default=100, type=int,
                        help='random patch number for training, and the rest of patches are for eval')

    # Paths
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser


def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # load data
    video_dir = os.path.join(args.dataset_dir)
    video_list = os.listdir(video_dir)
    video_list.sort(key=lambda x: int(x.split('.')[0]))

    # save path
    save_dir = args.output_dir
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)

    # crop patch
    patch_size = eval(args.patch_size)  # (T, H, W)
    n_patch_all = args.n_patch_all  # patch number of each whole video

    for i in range(len(video_list)):
        print(f"********* Video {i + 1} begins! *********")

        if i < args.n_patch_train:
            mode = 'train'
        else:
            mode = 'val'
        video = tifffile.imread(os.path.join(video_dir, video_list[i]))[:, 5:-5, 5:-5]
        origin_size = video.shape  # (T, H, W)

        j = 0
        while j < n_patch_all:
            t_start = np.random.randint(0, origin_size[0] - patch_size[0] + 1)
            h_start = np.random.randint(0, origin_size[1] - patch_size[1] + 1)
            w_start = np.random.randint(0, origin_size[2] - patch_size[2] + 1)
            video_patch = video[t_start:(t_start + patch_size[0]),
                                h_start:(h_start + patch_size[1]),
                                w_start:(w_start + patch_size[2])]
            global_max = np.max(video)
            video_patch_max = np.max(video_patch, axis=0)  # (H, W)
            valid_num = np.sum(video_patch_max > 0.1 * global_max)
            if valid_num > 0.01 * patch_size[1] * patch_size[2]:
                j = j + 1
                tifffile.imwrite(os.path.join(save_dir, mode, '%d_%d.tiff' % (int(video_list[i].split('.')[0]), j)),
                                 video_patch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Crop Patches Script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
