"""SSFA pipeline orchestration for webapp parity.

Runs the original argparse BG + segmentation inference scripts in sequence:
1) BG infer
2) BG postprocessing
3) Segmentation infer
4) Segmentation postprocessing
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_stage_dir(stage_name: str) -> Path:
    primary = ROOT_DIR / "codes" / stage_name
    fallback = ROOT_DIR / "legacy" / "codes" / stage_name
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Unable to resolve stage directory for '{stage_name}'. "
        f"Checked: {primary} and {fallback}"
    )


BG_STAGE_DIR = _resolve_stage_dir("BG_Rejection")
SEG_STAGE_DIR = _resolve_stage_dir("Segmentation")

DEFAULT_BG_CKPT_DIR = ROOT_DIR / "output" / "BG_output"
DEFAULT_SEG_CKPT_DIR = ROOT_DIR / "output" / "SEG_output"

ProgressCallback = Callable[[float, str], None]


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    # Legacy scripts expect pre-2.6 torch.load default (weights_only=False).
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        stdout_tail = "\n".join(proc.stdout.splitlines()[-80:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-80:])
        raise RuntimeError(
            "Pipeline command failed\n"
            f"cwd: {cwd}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout (tail):\n{stdout_tail}\n"
            f"stderr (tail):\n{stderr_tail}"
        )


def _write_manifest(artifacts: dict[str, str], extra: dict | None = None) -> str:
    """Write or update the manifest JSON for a pipeline run."""
    run_dir = Path(artifacts["run_dir"]).resolve()
    manifest_path = Path(artifacts["manifest_json"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _rel(path_str: str) -> str:
        path = Path(path_str).resolve()
        try:
            return str(path.relative_to(run_dir))
        except ValueError:
            return str(path)

    stage_files = {
        "input_dataset_tiff": _rel(artifacts["input_dataset_tiff"]),
        "background_tiff": _rel(artifacts["stage_background_tiff"]),
        "bg_rejected_wo_postprocessing_tiff": _rel(
            artifacts["stage_bg_rejected_wo_postprocessing_tiff"]
        ),
        "bg_rejected_tiff": _rel(artifacts["stage_bg_rejected_tiff"]),
        "enc_feats_tiff": _rel(artifacts["stage_enc_feats_tiff"]),
        "latent_tiff": _rel(artifacts["stage_latent_tiff"]),
        "segmentation_tiff": _rel(artifacts["stage_segmentation_tiff"]),
        "temporal_traces_mat": _rel(artifacts["stage_temporal_traces_mat"]),
    }

    quick_access = {
        "input_tiff": _rel(artifacts["input_tiff"]),
        "background_tiff": _rel(artifacts["background_tiff"]),
        "bg_rejected_tiff": _rel(artifacts["bg_rejected_tiff"]),
        "segmentation_tiff": _rel(artifacts["segmentation_tiff"]),
        "temporal_traces_mat": _rel(artifacts["temporal_traces_mat"]),
    }

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "legacy_output_dir": _rel(artifacts["legacy_output_dir"]),
        "stage_files": stage_files,
        "quick_access": quick_access,
        "metadata": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
        },
    }

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
            if isinstance(existing, dict):
                metadata = existing.get("metadata", {})
                if isinstance(metadata, dict):
                    manifest["metadata"].update(metadata)
        except json.JSONDecodeError:
            pass

    if extra:
        manifest["metadata"].update(extra)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return str(manifest_path)


def write_pipeline_manifest(artifacts: dict[str, str], extra: dict | None = None) -> str:
    """Public helper for updating manifest with additional metadata."""
    return _write_manifest(artifacts, extra=extra)


def save_numbered_tiff(volume: np.ndarray, dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    input_tiff = dataset_dir / "1.tiff"
    tifffile.imwrite(str(input_tiff), volume.astype(np.uint16))
    return input_tiff


def _convert_checkpoint_for_single_gpu(src_ckpt: Path, dst_ckpt: Path) -> None:
    """Strip DataParallel module prefixes when needed for inference scripts."""
    try:
        ckpt = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(src_ckpt), map_location="cpu")
    if not isinstance(ckpt, dict) or "net" not in ckpt or not isinstance(ckpt["net"], dict):
        shutil.copy2(src_ckpt, dst_ckpt)
        return

    state = ckpt["net"]
    if state and all(k.startswith("module.") for k in state.keys()):
        ckpt["net"] = {k.replace("module.", "", 1): v for k, v in state.items()}

    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(dst_ckpt))


def run_legacy_pipeline(
    volume: np.ndarray,
    run_dir: str | Path,
    device_idx: int = 0,
    device_type: str | None = None,
    batch_size: int = 1,
    bg_ckpt_dir: str | Path = DEFAULT_BG_CKPT_DIR,
    seg_ckpt_dir: str | Path = DEFAULT_SEG_CKPT_DIR,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    """Execute SSFA full pipeline and return artifact file paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    bg_ckpt_dir = Path(bg_ckpt_dir)
    seg_ckpt_dir = Path(seg_ckpt_dir)
    _ensure_file(bg_ckpt_dir / "checkpoint.pth", "BG checkpoint")
    _ensure_file(seg_ckpt_dir / "checkpoint.pth", "Segmentation checkpoint")

    input_dataset = run_dir / "input_dataset"
    legacy_output = run_dir / "legacy_output"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    converted_bg_ckpt = run_dir / "converted_ckpts" / "bg" / "checkpoint.pth"
    converted_seg_ckpt = run_dir / "converted_ckpts" / "seg" / "checkpoint.pth"

    _convert_checkpoint_for_single_gpu(bg_ckpt_dir / "checkpoint.pth", converted_bg_ckpt)
    _convert_checkpoint_for_single_gpu(seg_ckpt_dir / "checkpoint.pth", converted_seg_ckpt)

    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = device_type.lower()
    if device_type not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device_type: {device_type}")

    gpu_ids = str(device_idx) if device_type == "cuda" else ""

    if progress_callback:
        progress_callback(0.05, "Preparing input TIFF stack")

    input_tiff = save_numbered_tiff(volume, input_dataset)

    # Stage 1: BG infer
    if progress_callback:
        progress_callback(0.20, "Running background inference")

    bg_infer_cmd = [
        sys.executable,
        "infer.py",
        "--batch_size",
        str(batch_size),
        "--wo_mean",
        "--patch_size",
        "(128, 128, 128)",
        "--stride_size",
        "(128, 128, 128)",
        "--gsize",
        "8",
        "--dataset_dir",
        str(input_dataset),
        "--output_dir",
        str(legacy_output),
        "--ckpt_dir",
        str(converted_bg_ckpt.parent),
        "--device",
        device_type,
        "--gpu_ids",
        gpu_ids,
    ]
    _run_cmd(bg_infer_cmd, BG_STAGE_DIR)

    # Stage 2: BG postprocess
    if progress_callback:
        progress_callback(0.45, "Running background postprocessing")

    bg_post_cmd = [
        sys.executable,
        "postprocessing_discard.py",
        "--dataset_name",
        "rmbg",
        "--dataset_dir",
        str(legacy_output),
    ]
    _run_cmd(bg_post_cmd, BG_STAGE_DIR)

    # Stage 3: Seg infer
    if progress_callback:
        progress_callback(0.60, "Running segmentation inference")

    seg_infer_cmd = [
        sys.executable,
        "infer.py",
        "--batch_size",
        str(batch_size),
        "--wo_min",
        "--normalize_factor",
        "1",
        "--patch_size",
        "(128, 128, 128)",
        "--stride_size",
        "(128, 128, 128)",
        "--latent_patch_size",
        "(128, 32, 32)",
        "--latent_stride_size",
        "(128, 32, 32)",
        "--training_mode",
        "joint",
        "--fps",
        "15",
        "--gsize",
        "8",
        "--ext_rate",
        "292.3",
        "--dataset_dir",
        str(legacy_output / "rmbg"),
        "--output_dir",
        str(legacy_output),
        "--ckpt_dir",
        str(converted_seg_ckpt.parent),
        "--device",
        device_type,
        "--gpu_ids",
        gpu_ids,
    ]
    _run_cmd(seg_infer_cmd, SEG_STAGE_DIR)

    # Stage 4: Seg postprocess
    if progress_callback:
        progress_callback(0.85, "Running segmentation postprocessing")

    seg_post_cmd = [
        sys.executable,
        "postprocessing.py",
        "--gsize",
        "8",
        "--dataset_name",
        "",
        "--rmbg_dir",
        str(legacy_output / "rmbg"),
        "--latent_dir",
        str(legacy_output / "latent"),
        "--output_dir",
        str(legacy_output),
    ]
    _run_cmd(seg_post_cmd, SEG_STAGE_DIR)

    if progress_callback:
        progress_callback(0.95, "Collecting artifacts")

    bg_tiff = legacy_output / "bg" / "1.tiff"
    rmbg_wo_post_tiff = legacy_output / "rmbg_wo_postprocessing" / "1.tiff"
    rmbg_tiff = legacy_output / "rmbg" / "1.tiff"
    enc_feats_tiff = legacy_output / "enc_feats" / "1.tiff"
    latent_tiff = legacy_output / "latent" / "1.tiff"
    seg_tiff = legacy_output / "seg" / "1.tiff"
    sig_mat = legacy_output / "sig" / "1.mat"

    _ensure_file(bg_tiff, "Background TIFF")
    _ensure_file(rmbg_wo_post_tiff, "BG-rejected (pre-postprocessing) TIFF")
    _ensure_file(rmbg_tiff, "BG-rejected TIFF")
    _ensure_file(enc_feats_tiff, "Encoded feature TIFF")
    _ensure_file(latent_tiff, "Latent TIFF")
    _ensure_file(seg_tiff, "Segmentation TIFF")
    _ensure_file(sig_mat, "Temporal traces MAT")

    artifacts = {
        "run_dir": str(run_dir),
        "legacy_output_dir": str(legacy_output),
        "input_dataset_tiff": str(input_tiff),
        "stage_background_tiff": str(bg_tiff),
        "stage_bg_rejected_wo_postprocessing_tiff": str(rmbg_wo_post_tiff),
        "stage_bg_rejected_tiff": str(rmbg_tiff),
        "stage_enc_feats_tiff": str(enc_feats_tiff),
        "stage_latent_tiff": str(latent_tiff),
        "stage_segmentation_tiff": str(seg_tiff),
        "stage_temporal_traces_mat": str(sig_mat),
        "input_tiff": str(artifacts_dir / "input.tiff"),
        "background_tiff": str(artifacts_dir / "background.tiff"),
        "bg_rejected_tiff": str(artifacts_dir / "bg_rejected.tiff"),
        "segmentation_tiff": str(artifacts_dir / "segmentation_mask.tiff"),
        "temporal_traces_mat": str(artifacts_dir / "temporal_traces.mat"),
        "manifest_json": str(artifacts_dir / "manifest.json"),
    }

    shutil.copy2(input_tiff, artifacts["input_tiff"])
    shutil.copy2(bg_tiff, artifacts["background_tiff"])
    shutil.copy2(rmbg_tiff, artifacts["bg_rejected_tiff"])
    shutil.copy2(seg_tiff, artifacts["segmentation_tiff"])
    shutil.copy2(sig_mat, artifacts["temporal_traces_mat"])
    write_pipeline_manifest(artifacts)

    if progress_callback:
        progress_callback(1.0, "Pipeline complete")

    return artifacts
