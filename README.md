# DeepDefinite

DeepDefinite is a universally applicable algorithm for neuron extraction. Its description and benchmarking can be found in [(##############)](http://www.nature.com/articles/ncomms12190).



---

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

We provided our dataset here. Please download the dataset and put them into the folder `/dataset`. Of course, you can put your dataset in it. We assume the directory structure should be as follows.

```
dataset
├── train
│   ├── 1.tiff
│   ├── 2.tiff
│   └── ...
├── test
│   ├── 1.tiff
│   ├── 2.tiff
│   └── ...
└── val
    ├── 1.tiff
    ├── 2.tiff
    └── ...
```

There are two folders named `/train` and `/test`. Each folder contains many tif or tiff files.

### Prepare the pretrained weights

If you just want to have a try and skip the training process, please put our pretrained parameters into the folder `/output/BG_output/` and `/output/SEG_output/`



---

## **Remove the background**

### Training

Training the background removal network can be performed using 

```bash
python -u ./codes/BG_Rejection/main.py --wo_mean --gsize 7 --dataset_dir ./dataset --output_dir ./output/BG_output --gpu_ids 0,1,2,3
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
python -u ./codes/BG_Rejection/infer.py --wo_mean --gsize 7 --dataset_dir ./dataset/test --output_dir ./output/BG_output --ckpt_dir ./output/BG_output --patch_size '(128, 128, 128)' --stride_size '(96, 96, 96)' --gpu_ids 0,1,2,3
```

The following parameters can be configured in the command.

```python
def get_args_parser():
    parser = argparse.ArgumentParser('Full Image inference', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--wo_mean', action='store_true')

    parser.add_argument('--patch_size', default='', type=str)
    parser.add_argument('--stride_size', default='', type=str)

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

### Postprocessing

If you still think the video is not clean enough after removing the background using the command above, please run

```bash
python -u ./codes/BG_Rejection/postprocessing_discard.py --dataset_dir ./output/BG_output --threshold_max 0.5
```

This command will perform thresholding on the processed videos. 



---

## Segmentation

### Training

Training the background removal network can be performed using 

```bash
python -u ./codes/Segmentation/main.py --wo_min --training_mode joint --gsize 7 --fps 15 --dataset_dir ./datasets --output_dir ./output/SEG_output --gpu_ids 0,1,2,3
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
python -u ./codes/Segmentation/infer.py --wo_min --fps 15 --gsize 7 --dataset_dir ./output/BG_output/rmbg --output_dir ./output/SEG_output --ckpt_dir ./output/SEG_output --patch_size '(128, 128, 128)' --stride_size '(96, 96, 96)' --latent_patch_size '(128, 32, 32)' --latent_stride_size '(96, 24, 24)' --gpu_ids 0,1
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

### Postprocessing