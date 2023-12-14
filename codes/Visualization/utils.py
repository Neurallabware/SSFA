import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from scipy import ndimage
from scipy.ndimage import binary_dilation, generate_binary_structure


def compute_corr(a, b):
    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    a_mean = a_t - torch.mean(a_t, dim=0, keepdim=True)
    b_mean = b_t - torch.mean(b_t, dim=0, keepdim=True)
    a_n = F.normalize(a_mean, dim=0)
    b_n = F.normalize(b_mean, dim=0)
    corr = torch.einsum('tm,tn->mn', a_n, b_n)
    return corr.numpy()


def extract_signal(feat, mask):
    feat_2d = feat.reshape(feat.shape[0], -1)
    masked_feat = feat_2d[:, mask.flatten()]
    return masked_feat


def array_intersect_with_indices(a, b):
    s = {tuple(x) for x in a}
    common_elements = [x for x in b if tuple(x) in s]
    common_indices = [np.where(np.all(a == element, axis=1))[0][0] for element in common_elements]
    return np.array(common_elements), np.array(common_indices)


def gen_struct_by_R(R):
    r = np.ceil(R).astype(int)
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    distances = np.sqrt(x ** 2 + y ** 2)
    in_neighborhood = distances <= R
    outer_ring_has_true = np.any(in_neighborhood[0]) or np.any(in_neighborhood[-1]) or np.any(
        in_neighborhood[:, 0]) or np.any(in_neighborhood[:, -1])
    if not outer_ring_has_true:
        in_neighborhood = in_neighborhood[1:-1, 1:-1]
    return in_neighborhood


def get_outer_pixels(mask_array, struct):
    dilated_mask = binary_dilation(mask_array, structure=struct, iterations=1).astype(int)
    dilated_mask = dilated_mask - mask_array
    outer_pixels = np.column_stack(np.where(dilated_mask == 1))
    return outer_pixels


def update_masks(feat, masks, struct, corr_thre=0.1, max_iterations=1000):
    updated_masks = masks.copy().astype('float32')
    total_mask = np.any(masks, axis=0)
    total_mask = ~total_mask
    xs, ys = np.where(total_mask)
    pixels = extract_signal(feat, total_mask)

    mean_feats = []
    for mask in masks:
        ex_feats = extract_signal(feat, mask)
        mean_feat = np.mean(ex_feats, axis=-1)
        mean_feats.append(mean_feat)

    mean_feats = np.stack(mean_feats, axis=-1)
    pcorr_mat = compute_corr(mean_feats, pixels)
    not_noise = np.any(pcorr_mat > corr_thre, axis=0)

    pcorr_mat = pcorr_mat[:, not_noise]
    xs = xs[not_noise]
    ys = ys[not_noise]
    corr_indices = np.ones_like(xs) * (-1)
    coords = np.stack([xs, ys], axis=-1)

    for n in range(max_iterations):
        update_flag = False
        for i in range(pcorr_mat.shape[0]):
            mask = updated_masks[i]
            outer_pixels = get_outer_pixels(mask, struct)

            corr = pcorr_mat[i]
            condition = (corr_indices == -1) & (corr > corr_thre)

            candidate_indices = np.where(condition)[0]
            candidate_pixels = coords[candidate_indices, :]
            common_elements, common_indices = array_intersect_with_indices(candidate_pixels, outer_pixels)
            if common_elements.size != 0:
                total_idx = candidate_indices[common_indices]
                corr_indices[total_idx] = i
                mask[common_elements[:, 0], common_elements[:, 1]] = corr[total_idx]
                update_flag = True

        if not update_flag:
            print(f'Finished after running {n} iters!')
            break

    idx = np.where(corr_indices == -1)[0]
    not_locked_pixels = coords[idx, :]
    not_locked_mask = np.zeros_like(total_mask)
    not_locked_mask[not_locked_pixels[:, 0], not_locked_pixels[:, 1]] = True

    return updated_masks, not_locked_mask, corr_indices
