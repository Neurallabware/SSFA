import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
import argparse
from utils import gen_struct_by_R, update_masks


def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)

    # Params
    parser.add_argument('--corr_thre', default=0.5, type=float, help='Correlation coefficient threshold')
    parser.add_argument('--R', default=2, type=int, help='Radius of struct using to grow footprints. In general, the larger the `R`, the larger the `corr_thre` should be.')
    parser.add_argument('--max_iter', default=100, type=int, help='Max iterations for growing footprints')

    # Paths
    parser.add_argument('--rmbg_dir', default='./output/BG_output/res_videos/rmbg', type=str)
    parser.add_argument('--seg_dir', default='./output/FINAL_output/seg', type=str)
    parser.add_argument('--output_dir', default='./output/FINAL_output/footprints', type=str)
    return parser


def main(args):
    ###############################################
    # Set params
    ###############################################
    corr_thre = args.corr_thre
    R = args.R
    max_iterations = args.max_iter

    rmbg_dir = args.rmbg_dir
    seg_dir = args.seg_dir
    SAVE = args.output_dir

    ###############################################
    # Main
    ###############################################
    for filename in os.listdir(rmbg_dir):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            file_rmbg_path = os.path.join(rmbg_dir, filename)
            file_seg_path = os.path.join(seg_dir, filename)
            
            if os.path.exists(file_seg_path):
                print()
                print(f'**************** {filename} starts! ****************')
                file_rmbg = tiff.imread(file_rmbg_path)
                file_seg = tiff.imread(file_seg_path)

                feat = file_rmbg
                feat = feat.astype('float32')
                masks = file_seg
                masks = masks.astype(bool)

                struct = gen_struct_by_R(R)
                # print('`struct` defining outer pixels:')
                # print(struct)

                res_path = os.path.join(SAVE, filename)
                updated_masks, _, corr_indices = update_masks(feat, masks, struct, corr_thre=corr_thre, max_iterations=max_iterations)

                tiff.imwrite(res_path, updated_masks.astype('float32'))
                print(f'The mask of contours saved at {res_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Contours Script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)


