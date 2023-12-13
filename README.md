# **DeepDefinite**

DeepDefinite is a universally applicable algorithm for neuron extraction. DeepDefinite is self-supervised and performs well. Its detailed description and benchmarking can be found in [(##############)](http://www.nature.com/articles/ncomms12190).

![SSFA_logo_long](SSFA_logo_long.svg)

- [**DeepDefinite**](#deepdefinite)
  - [**Getting Started**](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Install](#install)
    - [Prepare the dataset](#prepare-the-dataset)
    - [Prepare the pretrained weights](#prepare-the-pretrained-weights)
  - [**Remove the background**](#remove-the-background)
    - [Training](#training)
    - [Background removal](#background-removal)
  - [**Segmentation**](#segmentation)
    - [Training](#training-1)
    - [Segmentation](#segmentation-1)

## **Getting Started**

### Prerequisites

- Python 3
- NVIDIA GPU + CUDA CuDNN

Note: Perhaps users can run the codes we provided using only the CPU, but using CUDA is highly recommended.

### Install

- Clone the repository:

```bash
git clone https://github.com/jerryniu520126/DeepDefinite.git
cd DeepDefinite
```

- Dependencies: We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). The codes are tested under `PyTorch 1.9.1` and `CUDA 11.3`. Please install them under the instructions [here](https://pytorch.org/get-started/previous-versions/#v191). All the required packages are listed in `requirements.txt`.

### Prepare the dataset

We provided our dataset here. Please download the dataset and put them into the folder `/datasets/dataset_orig`. Of course, you can put your dataset in it. We assume the directory structure should be as follows.

```
datasets
└── dataset_orig
    ├── 1.tiff
    ├── 2.tiff
    └── ...
```

### Prepare the pretrained weights

If you just want to have a try and skip the training process, please put our pretrained parameters into the folder `/output/BG_output/` and `/output/SEG_output/`

## **Remove the background**

### Training

Perhaps, the input videos are very large and thus GPU does not have enough space to store. In this case, we recommend to run

```bash
python -u ./codes/BG_Rejection/crop.py \
--patch_size '(64, 256, 256)' \
--n_patch_all 150 \
--dataset_dir ./datasets/dataset_orig \
--output_dir ./datasets/dataset_in_patches_BG
```

The command will randomly crop patches from the original videos. This step is **optional** but **highly recommended**. The following parameters can be configured in the command.

```python
def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)
    # Patches
    parser.add_argument('--patch_size', default='(64, 256, 256)', type=str, help='(T, H, W)')
    parser.add_argument('--n_patch_all', default=150, type=int, help='random patch number of each whole video')
    parser.add_argument('--n_patch_train', default=999999, type=int,
                        help="random patch number for training, and the rest of patches are for eval."
                             "However, we usually recommend to put all patches to `train` in the BG_Rejection stage")

    # Paths
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser
```

Then, we can train the background removal network using 

```bash
python -u ./codes/BG_Rejection/main.py \
--wo_mean \
--gsize 8 \
--dataset_dir ./datasets/dataset_orig \
--output_dir ./output/BG_output \
--gpu_ids 0,1,2,3
```

The following parameters can be configured in the command.

```python
def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=10, type=int, help='every 10 epochs drop lr')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--wo_mean', action='store_true')

    # Model parameters
    parser.add_argument('--gsize', default=8, type=int, help='neuron radius')
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--f_maps', default=32, type=int, help='initial channels of feature maps')

    # Loss coefficients
    parser.add_argument('--L1_weight', default=1.0, type=float)
    parser.add_argument('--L2_weight', default=1.0, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_name', default='', type=str)
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluate patch results in test set')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--gpu_ids', default='0, 1, 2, 3', type=str, 
                        help='If GPU is used for training, specify the gpu ids in your machine')
    return parser
```

### Background removal

Inference (i.e. background removal on given videos) can be performed using

```bash
python -u ./codes/BG_Rejection/infer.py \
--wo_mean \
--gsize 8 \
--dataset_dir ./datasets/dataset_orig \
--output_dir ./output/BG_output/res_videos \
--ckpt_dir ./output/BG_output \
--patch_size '(128, 128, 128)' \
--stride_size '(96, 96, 96)' \
--gpu_ids 0,1,2,3
```

The following parameters can be configured in the command.

```python
def get_args_parser():
    parser = argparse.ArgumentParser('Full Image inference', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--wo_mean', action='store_true')

    parser.add_argument('--patch_size', default='(128, 128, 128)', type=str)
    parser.add_argument('--stride_size', default='(96, 96, 96)', type=str)

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
```

If users still think the video is not clean enough after removing the background using the command above, please run

```bash
python -u ./codes/BG_Rejection/postprocessing_discard.py \
--dataset_dir ./output/BG_output/res_videos \
--threshold_max 0.5
```

This command will perform thresholding on the processed videos. 



## **Segmentation**

### Training

Similarly, the input videos may be too large to put into GPU as a whole. In this case, we recommend to run

```bash
python -u ./codes/Segmentation/crop.py \
--patch_size '(128, 128, 128)' \
--n_patch_all 150 \
--dataset_dir ./output/BG_output/res_videos/rmbg \
--output_dir ./datasets/dataset_in_patches_SEG
```

The command will randomly crop patches from the original videos. Again, this step is **optional** but **highly recommended**. The following parameters can be configured in the command.

```python
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
```

Then, we can train the neuron segmentation network using 

```bash
python -u ./codes/Segmentation/main.py \
--wo_min \
--training_mode joint \
--gsize 8 \
--fps 15 \
--dataset_dir ./datasets/dataset_in_patches_SEG \
--output_dir ./output/SEG_output \
--gpu_ids 0,1,2,3
```

The following parameters can be configured in the command.

```python
def get_args_parser():
    parser = argparse.ArgumentParser('Hyper-Parameters Configuration', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--wo_min', action='store_true')
    parser.add_argument('--normalize_factor', default=1, type=int)

    # Model parameters
    parser.add_argument('--training_mode', default='joint', type=str)
    parser.add_argument('--gsize', default=5, type=int, help='neuron size')
    parser.add_argument('--fps', default=10, type=int, help='frame rate')
    parser.add_argument('--ext_rate', default=292.3, type=float, help='temporal extrusion rate')
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--f_maps', default=32, type=int, help='initial channels of feature maps')
    parser.add_argument('--n_fmaps', default=3, type=int, help='feature map levels')

    # Loss coefficients
    parser.add_argument('--L1_weight', default=1.0, type=float)
    parser.add_argument('--L2_weight', default=0.0, type=float)
    parser.add_argument('--Sparse_weight', default=10.0, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_name', default='', type=str)
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluate patch results in test set')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--gpu_ids', default='0, 1, 2, 3', type=str,
                        help='If GPU is used for training, specify the gpu ids in your machine')
    return parser
```

### Segmentation

Inference (i.e. segmentation on given videos) can be performed using

```bash
python -u ./codes/Segmentation/infer.py \
--wo_min \
--fps 15 \
--gsize 8 \
--dataset_dir ./output/BG_output/res_videos/rmbg \
--output_dir ./output/SEG_output/res_videos \
--ckpt_dir ./output/SEG_output \
--patch_size '(128, 128, 128)' \
--stride_size '(96, 96, 96)' \
--latent_patch_size '(128, 32, 32)' \
--latent_stride_size '(96, 24, 24)' \
--gpu_ids 0,1
```

The following parameters can be configured in the command.

```python
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
```

Finally, postprocess the results by running

```bash
python -u ./codes/Segmentation/postprocessing.py \
--gsize 8 \
--rmbg_dir ./output/BG_output/res_videos/rmbg \
--latent_dir ./output/SEG_output/res_videos/latent \
--output_dir ./output/FINAL_output
```

The following parameters can be configured in the command.

```python
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
```

In the end, the final segmentation masks and extracted calcium signals will be stored in `./output/FINAL_output`.

### Dataset, website and plugin
The MultiModel NeuralImaging Benchmark (MMNIB) dataset at https://drive.google.com/drive/folders/1095PivRqH5rDVSg9dDG2ixeqUUNveGmw?usp=drive_link
To get a live preview of the dataset or process your data online, please visit our webapp: http://region-31.seetacloud.com:36894/
The deepdefinite pipeline can also be used with a Fiji plugin: https://drive.google.com/drive/folders/1RCdgfLWDkRIgK-xEiY2Nl4R29YXpqVAB?usp=drive_link
