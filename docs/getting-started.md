# Getting started

## Install

The repository environment specifies Python 3.11:

```bash
conda env create -f environment.yaml
conda activate ssfa
```

For a pip-managed environment, use `requirements.txt` and choose PyTorch/CUDA for the target machine. The docs environment is separate.

## Prepare data and checkpoints

Place numerically named TIFF stacks under `datasets/dataset_orig/`. The inference scripts require at least 128 frames for temporal patching. Provide compatible background and segmentation checkpoints; defaults resolve under `output/BG_output` and `output/SEG_output`. Check `configs/paths/default.yaml`.

## Inspect configuration

```bash
python src/pipeline.py task=infer_full --cfg job
```

## Run inference

```bash
python src/pipeline.py task=infer_full \
  pipeline.input_dir=/path/to/tiff_stacks \
  pipeline.output_dir=/path/to/ssfa_results
```

This command is documented from source; the documentation build does not execute it. The pipeline config defaults to CUDA.

## Web UI

```bash
python src/webui.py webui.port=7860 webui.device_type=auto
```

See the root README for Fiji/DeepImageJ plugin installation and Git LFS artifacts.
