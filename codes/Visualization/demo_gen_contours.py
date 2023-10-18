import tifffile as tiff
import scipy.io
import os

BASE = '/data/yuwang/yuwang_server_backup/2p_wd_data/macroscope/2.5x_rasgrf148d_multisite/11242022/1'
RES = 'network_discard_top1_thres0.5'

# 使用 os.path.join 创建文件路径
file_2p_path = os.path.join(BASE, "2p.tiff")
file_wd_path = os.path.join(BASE, "wd.tiff")
file_rmbg_path = os.path.join(BASE, RES, "rmbg.tiff")
file_seg_path = os.path.join(BASE, RES, "seg.tiff")
file_sig_mat_path = os.path.join(BASE, RES, "sig.mat")

# 1. 读取BASE/2p.tiff
file_2p = tiff.imread(file_2p_path)

# 2. 读取BASE/wd.tiff
file_wd = tiff.imread(file_wd_path)

# 3. 读取BASE/RES/rmbg.tiff
file_rmbg = tiff.imread(file_rmbg_path)

# 4. 读取BASE/RES/seg.tiff
file_seg = tiff.imread(file_seg_path)

# 5. 读取BASE/RES/sig.mat
file_sig = scipy.io.loadmat(file_sig_mat_path)

# 现在，你可以使用这些变量来访问相应的文件数据
# ---------------------------------------------------------------------------
import sys
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
    # 计算邻域大小
    r = np.ceil(R).astype(int)

    # 创建一个二维坐标网格
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))

    # 计算每个点到中心点的距离
    distances = np.sqrt(x ** 2 + y ** 2)

    # 创建布尔数组，表示是否在 R-邻域内
    in_neighborhood = distances <= R

    # 检查最外围一圈是否存在 True
    outer_ring_has_true = np.any(in_neighborhood[0]) or np.any(in_neighborhood[-1]) or np.any(
        in_neighborhood[:, 0]) or np.any(in_neighborhood[:, -1])

    # 移除最外围一圈
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
    # print('pcorr_mat.shape', pcorr_mat.shape)

    for n in range(max_iterations):
        update_flag = False
        for i in range(pcorr_mat.shape[0]):
            mask = updated_masks[i]
            outer_pixels = get_outer_pixels(mask, struct)

            # Check if any of the outer pixels have already been assigned to a mask (locked)
            corr = pcorr_mat[i]
            condition = (corr_indices == -1) & (corr > corr_thre)
            # print('corr[condition].shape:', corr[condition].shape)
            # print()

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


###############################################
# Set params
###############################################
corr_thres = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
R = 4  # 一般情况下，R越大，corr_thre就要越大，否则可能会容纳进一些奇怪的pixels到mask中
max_iterations = 100

SAVE = os.path.join(BASE, RES, '001 - contours (by yujia)', f'dilate_R{R}')
os.makedirs(SAVE, exist_ok=True)


###############################################
# Main
###############################################
feat = file_rmbg
feat = feat.astype('float32')
masks = file_seg
masks = masks.astype(bool)

struct = gen_struct_by_R(R)
print('`struct` defining outer pixels:')
print(struct)

for corr_thre in corr_thres:
    res_path = os.path.join(SAVE, f'Masks_{corr_thre}_dilateR{R}.tif')
    not_path = os.path.join(SAVE, f'not_matched_{corr_thre}_dilateR{R}.tif')
    updated_masks, not_locked_mask, corr_indices = update_masks(feat, masks, struct,
                                                                corr_thre=corr_thre,
                                                                max_iterations=max_iterations)

    tiff.imwrite(res_path, updated_masks.astype('float32'))
    tiff.imwrite(not_path, not_locked_mask.astype('float32'))