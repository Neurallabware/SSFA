import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from utils import gen_struct_by_R, update_masks


###############################################
# Set params
###############################################
corr_thre = 0.5
R = 2  # In general, the larger the `R`, the larger the `corr_thre` should be. Otherwise, some strange pixels may be included.
max_iterations = 100

file_rmbg_path = './output/BG_output/res_videos/rmbg/1.tiff'
file_seg_path = './output/FINAL_output/seg/1.tiff'
SAVE = './output/FINAL_output/contours'



###############################################
# Main
###############################################
os.makedirs(SAVE, exist_ok=True)
file_rmbg = tiff.imread(file_rmbg_path)
file_seg = tiff.imread(file_seg_path)

feat = file_rmbg
feat = feat.astype('float32')
masks = file_seg
masks = masks.astype(bool)

struct = gen_struct_by_R(R)
# print('`struct` defining outer pixels:')
# print(struct)

res_path = os.path.join(SAVE, f'Masks_{corr_thre}_dilateR{R}.tif')
updated_masks, _, corr_indices = update_masks(feat, masks, struct, corr_thre=corr_thre, max_iterations=max_iterations)

tiff.imwrite(res_path, updated_masks.astype('float32'))
print(f'The mask of contours saved at {res_path}')