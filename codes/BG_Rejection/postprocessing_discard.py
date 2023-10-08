import os
import cv2
import math
import time
import argparse
import tifffile
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)

    parser.add_argument('--dataset_name', default='rmbg', type=str)
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--threshold_max', default=0.5, type=float)
    return parser


def erode_dilate(mask):
    kernel_erode = np.ones((5, 5), dtype=np.uint8)
    kernel_dilate = np.ones((7, 7), dtype=np.uint8)
    # erode and dilate
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.erode(mask, kernel=kernel_erode, iterations=1)
    mask = cv2.dilate(mask, kernel=kernel_dilate, iterations=1)
    mask = mask > 0
    return mask


def main(args):
    video_path = os.path.join(args.dataset_dir, 'rmbg_wo_postprocessing')
    save_path = os.path.join(args.dataset_dir, args.dataset_name)
    os.makedirs(save_path, exist_ok=True)

    video_list = os.listdir(video_path)
    video_list.sort(key=lambda x: int(x.split('.')[0]))
    for i in range(len(video_list)):
        print(i + 1)
        start_time = time.time()
        video = tifffile.imread(os.path.join(video_path, video_list[i])).astype(np.float32)
        # if 'drosophila' in args.dataset_dir:
        #     print(i + 1)
        #     video[:, :30, :] = 0
        #     video[:, -30:, :] = 0
        #     video[:, :, :30] = 0
        #     video[:, :, -30:] = 0
        video_max_proj = np.max(video, axis=0)

        # discard top 1%
        video_max_proj_new = np.sort(video_max_proj.flatten())
        L = math.ceil(len(video_max_proj_new) * 0.99)
        video_max_proj_new = video_max_proj_new[:L]

        mask = video_max_proj > np.max(video_max_proj_new) * args.threshold_max
        mask = erode_dilate(mask)

        out = video * mask
        tifffile.imwrite(os.path.join(save_path, video_list[i]), out.astype(np.uint16))

        end_time = time.time()
        time_step = end_time - start_time
        print('Time cost of background rejection postprocessing: %.4f' % time_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Background Rejection Model Postprocessing Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
