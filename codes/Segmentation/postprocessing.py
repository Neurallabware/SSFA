import os
import math
import argparse
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.io import savemat
from scipy.cluster.hierarchy import linkage, fcluster


def temporal_dist(a, b):
    return 1 - np.corrcoef(a, b)[0, 1]


def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)

    parser.add_argument('--gsize', default=5, type=int)
    parser.add_argument('--dataset_name', default='', type=str)
    parser.add_argument('--rmbg_dir', default='', type=str)
    parser.add_argument('--latent_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--intensity_threshold', default=0.8, type=float)
    parser.add_argument('--temporal_threshold', default=0.3, type=float)
    return parser


def main(args):
    rmbg_list = os.listdir(args.rmbg_dir)
    latent_list = os.listdir(args.latent_dir)

    rmbg_list.sort(key=lambda x: int(x.split('.')[0]))
    latent_list.sort(key=lambda x: int(x.split('.')[0]))

    for index in range(len(rmbg_list)):
        print(index + 1)
        rmbg = tifffile.imread(os.path.join(args.rmbg_dir, rmbg_list[index])).astype(np.float32) / 65535
        latent = tifffile.imread(os.path.join(args.latent_dir, latent_list[index])).astype(np.float32) / 65535

        # remove edge and the last frame
        latent[:, 0, :] = 0
        latent[:, -1, :] = 0
        latent[:, :, 0] = 0
        latent[:, :, -1] = 0
        latent[-1, :, :] = 0

        # use 3*3 kernel to extract temporal signal
        kernel = np.ones((1, 1, 3, 3), dtype=np.float32)
        kernel = kernel / np.sum(kernel)  # normalization
        kernel = torch.from_numpy(kernel)

        rmbg = torch.from_numpy(rmbg).unsqueeze(1)
        rmbg_down = F.interpolate(rmbg, size=(latent.shape[1], latent.shape[2]), mode='bilinear', align_corners=True)
        rmbg_down = F.pad(rmbg_down, pad=(1, 1, 1, 1), mode='reflect')
        rmbg_down = F.conv2d(rmbg_down, kernel)
        rmbg_down = rmbg_down.clamp_(0, 1).squeeze(1).numpy()

        # max projection & signal intensity mask
        max_proj = np.max(latent, axis=0)
        mask_binary = max_proj > args.intensity_threshold

        # record position and temporal signal
        points_all = np.argwhere(mask_binary)

        if len(points_all) != 0:
            temporal_all = np.stack([rmbg_down[:, points_all[i][0], points_all[i][1]] for i in range(len(points_all))],
                                    axis=0)

            # filter invalid temporal signal
            keep = np.max(temporal_all, axis=1) != np.min(temporal_all, axis=1)
            points_all = points_all[keep]
            temporal_all = temporal_all[keep]
        else:
            points_all = np.array([[1, 1], [2, 2]])
            temporal_all = np.random.randn(2, rmbg_down.shape[0])

        if len(points_all) == 1:
            points_all = points_all.repeat(2, axis=0)
            temporal_all = temporal_all.repeat(2, axis=0)

        # spatial clustering
        z = linkage(points_all, method='single')
        c = fcluster(z, t=math.ceil(args.gsize / 2), criterion='distance')

        groups = []
        temporals = []
        centroids = []
        for i in range(1, max(c) + 1):
            group = points_all[np.squeeze(np.argwhere(c == i), axis=-1)]
            temporal = temporal_all[np.squeeze(np.argwhere(c == i), axis=-1)]

            if len(group) == 1:
                groups.append(group)
                temporals.append(np.squeeze(temporal))
                centroids.append(np.round(np.squeeze(group)).astype(np.uint16))
                continue

            # temporal clustering for each pixel group (according to temporal correlation within each group)
            z_i = linkage(temporal, method='average', metric=temporal_dist)
            c_i = fcluster(z_i, t=args.temporal_threshold, criterion='distance')

            for j in range(1, max(c_i) + 1):
                groups.append(group[np.squeeze(np.argwhere(c_i == j), axis=-1)])
                temporals.append(np.mean(temporal[np.squeeze(np.argwhere(c_i == j), axis=-1)], axis=0))
                centroids.append(
                    np.round(np.mean(group[np.squeeze(np.argwhere(c_i == j), axis=-1)], axis=0)).astype(np.uint16))
        centroids = np.stack(centroids, axis=0)
        temporals = np.stack(temporals)

        print('Neuron Num: %d' % len(groups))
        seg_map = torch.zeros((len(groups),) + rmbg_down.shape[1:], dtype=torch.float32).unsqueeze(0)
        for i in range(len(groups)):
            for j in range(len(groups[i])):
                seg_map[:, i, groups[i][j][0], groups[i][j][1]] = 1
        seg_map = F.interpolate(seg_map, size=(rmbg.shape[2], rmbg.shape[3])) > 0
        seg_map = (seg_map.squeeze().numpy() * 65535).astype(np.uint16)

        tifffile.imwrite(os.path.join(args.output_dir, 'seg' + args.dataset_name, rmbg_list[index]), seg_map)

        # save temporal signal
        sig_dict = {'temporal_sig': temporals}
        savemat(os.path.join(args.output_dir, 'sig' + args.dataset_name, rmbg_list[index].replace('.tiff', '.mat')),
                sig_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Segmentation Model Postprocessing Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, 'sig' + args.dataset_name), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'seg' + args.dataset_name), exist_ok=True)
    main(args)
