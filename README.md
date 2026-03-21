# SSFA

SSFA stands for **Self-Supervised Functional Analysis**.

SSFA is a self-supervised pipeline for neuron extraction on single-photon (1p) calcium videos. The repository contains:
- Original BG rejection + segmentation scripts in `codes/`
- A Hydra orchestration layer in `src/` and `configs/`
- A Gradio web UI in `webapp/`

![pipeline](readme_figs/pipeline.png)

## Quick Start

### 1. Environment

Using Conda:

```bash
conda env create -f environment.yaml
conda activate ssfa
```

Or with pip:

```bash
pip install -r requirements.txt
```

Optional runtime env defaults:

```bash
cp .env.example .env
```

### 2. Dataset layout

Place TIFF stacks in `datasets/dataset_orig` with numeric file names:

```text
datasets/
└── dataset_orig/
    ├── 1.tiff
    ├── 2.tiff
    └── ...
```

## Hydra Pipeline Interface

Main entrypoint:

```bash
python src/pipeline.py task=<task_name>
```

Supported tasks:
- `bg_crop`
- `bg_train`
- `bg_infer`
- `bg_post`
- `seg_crop`
- `seg_train`
- `seg_infer`
- `seg_post`
- `visualize`
- `infer_full`

### Common usage

End-to-end inference:

```bash
python src/pipeline.py \
  task=infer_full \
  pipeline.input_dir=./datasets/dataset_orig \
  pipeline.output_dir=./output/pipeline_output
```

BG training:

```bash
python src/pipeline.py task=bg_train bg.train.gsize=8 bg.train.epochs=50
```

Segmentation training:

```bash
python src/pipeline.py task=seg_train seg.train.gsize=8 seg.train.epochs=100
```

Inspect composed config:

```bash
python src/pipeline.py task=infer_full --cfg job
python src/pipeline.py task=bg_train --cfg job
python src/pipeline.py task=seg_train --cfg job
```

### Config groups

Hydra config groups exposed in `configs/`:
- `paths.*`
- `pipeline.*`
- `bg.*`
- `seg.*`
- `webui.*`
- `hydra.*`

### Checkpoint compatibility

The Hydra runner includes compatibility shims for modern PyTorch and legacy checkpoints:
- Forces legacy `torch.load(..., weights_only=False)` behavior via environment
- Auto-converts DataParallel checkpoints (`module.*`) into single-device compatible temporary checkpoints during inference

Defaults:
- BG checkpoint: `./output/BG_output/checkpoint.pth`
- Seg checkpoint: `./output/SEG_output/checkpoint.pth`

## Web UI

Hydra web UI entrypoint:

```bash
python src/webui.py webui.port=7860 webui.device=0 webui.device_type=auto
```

Then open:
- `http://localhost:7860`

The web UI accepts MP4/TIFF input, runs full SSFA inference, and exports:
- Full ZIP bundle
- Background TIFF
- Background-rejected TIFF
- Segmentation TIFF
- Temporal traces MAT
- Manifest JSON

`webui.device_type` options:
- `auto` (default): uses CUDA if available, otherwise CPU
- `cuda`
- `cpu`

### Direct launch script

```bash
./webapp/launch.sh 0
```

## Legacy Compatibility

Original scripts remain unchanged in `codes/` and can still be called directly, for example:

```bash
python -u ./codes/BG_Rejection/infer.py ...
python -u ./codes/Segmentation/infer.py ...
python -u ./codes/Segmentation/postprocessing.py ...
```

The Hydra layer is orchestration-only and preserves existing compute paths for parity.

## Outputs

For `task=infer_full`, default staged outputs are:
- `output/pipeline_output/bg_stage/`
- `output/pipeline_output/seg_stage/`
- `output/pipeline_output/final_output/`

## Notes

- Input videos for the inference scripts should contain at least 128 frames for temporal patching.
- Web UI runtime folders (`webapp/jobs/`, `webapp/jobs_full_test/`) are generated on demand and ignored by git.
- MMNIB dataset and Fiji plugin links from prior releases remain applicable.
