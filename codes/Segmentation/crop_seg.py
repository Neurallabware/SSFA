import os
import tifffile
import numpy as np

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# load data
root_path = '/data/yuwang/yuwang_server_backup/SSFA/Segmentation/datasets/2p_wd_data/widefield_microscope/drosophila'

video_dir = os.path.join(root_path, 'rmbg')
video_list = os.listdir(video_dir)
video_list.sort(key=lambda x: int(x.split('.')[0]))

# save path
save_dir = os.path.join(root_path, 'patch128')
os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)

# crop patch
patch_size = [128, 128, 128] # (T, H, W)
patch_num = 150 # patch number of each whole video

for i in range(len(video_list)):
    print(i + 1)
        
    if i < 100:
        mode = 'train'
    else:
        mode = 'val'
    video = tifffile.imread(os.path.join(video_dir, video_list[i]))[:, 5:-5, 5:-5]
    origin_size = video.shape # (T, H, W)
    
    j = 0
    while j < patch_num:
        t_start = np.random.randint(0, origin_size[0] - patch_size[0] + 1)
        h_start = np.random.randint(0, origin_size[1] - patch_size[1] + 1)
        w_start = np.random.randint(0, origin_size[2] - patch_size[2] + 1)
        video_patch = video[t_start:t_start+patch_size[0],
                            h_start:h_start+patch_size[1],
                            w_start:w_start+patch_size[2]]
        global_max = np.max(video)
        video_patch_max = np.max(video_patch, axis=0) # (H, W)
        valid_num = np.sum(video_patch_max > 0.1 * global_max)
        if valid_num > 0.01 * patch_size[1] * patch_size[2]:
            tifffile.imwrite(os.path.join(save_dir, mode, '%d_%d.tiff' % (int(video_list[i].split('.')[0]), j + 1)), video_patch)
            j = j + 1