"""Inference and parity helpers for SSFA webapp MP4/TIFF pipeline."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile
from matplotlib.figure import Figure
from scipy.io import loadmat

try:
    from .pipeline_legacy import (
        DEFAULT_BG_CKPT_DIR,
        DEFAULT_SEG_CKPT_DIR,
        run_legacy_pipeline,
        write_pipeline_manifest,
    )
except ImportError:
    from pipeline_legacy import (
        DEFAULT_BG_CKPT_DIR,
        DEFAULT_SEG_CKPT_DIR,
        run_legacy_pipeline,
        write_pipeline_manifest,
    )


def _to_uint16(volume: np.ndarray) -> np.ndarray:
    if volume.dtype == np.uint16:
        return volume
    if np.issubdtype(volume.dtype, np.integer):
        maxv = int(np.iinfo(volume.dtype).max)
        if maxv == 0:
            return np.zeros_like(volume, dtype=np.uint16)
        return np.round(volume.astype(np.float32) / maxv * 65535.0).clip(0, 65535).astype(np.uint16)

    # float types
    v = volume.astype(np.float32)
    if np.nanmax(v) <= 1.0:
        v = v * 65535.0
    return np.nan_to_num(v, nan=0.0, posinf=65535.0, neginf=0.0).clip(0, 65535).astype(np.uint16)


def decode_mp4_to_volume(mp4_path: str, max_frames: int | None = None) -> tuple[np.ndarray, dict]:
    """Decode MP4 to grayscale uint16 volume (T, H, W)."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open MP4: {mp4_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # 8-bit -> 16-bit preserving range proportionally.
            gray16 = (gray.astype(np.uint16) * 257).astype(np.uint16)
            frames.append(gray16)

            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from MP4: {mp4_path}")

    volume = np.stack(frames, axis=0)
    meta = {
        "frame_count": int(volume.shape[0]),
        "height": int(volume.shape[1]),
        "width": int(volume.shape[2]),
        "fps": fps,
        "source": "mp4",
    }
    return volume, meta


def load_input_volume(input_path: str, max_frames: int | None = None) -> tuple[np.ndarray, dict]:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".mp4":
        return decode_mp4_to_volume(str(path), max_frames=max_frames)

    if suffix not in {".tif", ".tiff"}:
        raise ValueError(f"Unsupported input type: {suffix}. Expected mp4/tif/tiff")

    arr = tifffile.imread(str(path))

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
        # (T, H, W, C) -> grayscale
        arr = np.mean(arr[..., :3], axis=-1)
    else:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}. Expected (T,H,W) or (H,W)")

    if max_frames is not None:
        arr = arr[:max_frames]

    vol = _to_uint16(arr)
    meta = {
        "frame_count": int(vol.shape[0]),
        "height": int(vol.shape[1]),
        "width": int(vol.shape[2]),
        "fps": None,
        "source": "tiff",
    }
    return vol, meta


def run_legacy_full_pipeline(
    input_path: str,
    run_dir: str,
    device_idx: int = 0,
    device_type: str | None = None,
    batch_size: int = 1,
    bg_ckpt_dir: str | os.PathLike[str] = DEFAULT_BG_CKPT_DIR,
    seg_ckpt_dir: str | os.PathLike[str] = DEFAULT_SEG_CKPT_DIR,
    max_frames: int | None = None,
    progress_callback=None,
) -> dict:
    """Decode input and run the full SSFA pipeline."""
    if progress_callback:
        progress_callback(0.02, "Decoding input")

    volume, input_meta = load_input_volume(input_path, max_frames=max_frames)

    if volume.shape[0] < 128:
        raise ValueError(
            f"Input has {volume.shape[0]} frames, but SSFA inference requires >=128 temporal frames"
        )
    if volume.shape[0] <= 128:
        # Legacy concat logic has an edge case at exactly one temporal patch.
        # Duplicate the last frame to ensure at least 2 temporal patch positions.
        volume = np.concatenate([volume, volume[-1:]], axis=0)
        input_meta["frame_count_padded"] = int(volume.shape[0])

    artifacts = run_legacy_pipeline(
        volume=volume,
        run_dir=run_dir,
        device_idx=device_idx,
        device_type=device_type,
        batch_size=batch_size,
        bg_ckpt_dir=bg_ckpt_dir,
        seg_ckpt_dir=seg_ckpt_dir,
        progress_callback=progress_callback,
    )

    bg = tifffile.imread(artifacts["background_tiff"]).astype(np.float32)
    neuron = tifffile.imread(artifacts["bg_rejected_tiff"]).astype(np.float32)
    inp = tifffile.imread(artifacts["input_tiff"]).astype(np.float32)

    recon = bg + neuron
    diff = np.abs(recon - inp)

    trace_meta = get_trace_metadata(artifacts)

    artifacts["input_meta_json"] = json.dumps(input_meta)
    artifacts["input_meta"] = input_meta
    artifacts["reconstruction_max_diff"] = float(diff.max())
    artifacts["reconstruction_mean_diff"] = float(diff.mean())
    artifacts.update(trace_meta)

    write_pipeline_manifest(
        artifacts,
        extra={
            "input_meta": input_meta,
            "reconstruction": {
                "max_diff": artifacts["reconstruction_max_diff"],
                "mean_diff": artifacts["reconstruction_mean_diff"],
            },
            "trace_metadata": trace_meta,
        },
    )

    return artifacts


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    lo = np.percentile(frame, 0.5)
    hi = np.percentile(frame, 99.5)
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.uint8)
    clipped = np.clip((frame.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    return clipped.astype(np.uint8)


def _read_tiff_frame(path: str, idx: int) -> np.ndarray:
    arr = tifffile.memmap(path)
    if arr.ndim == 2:
        if idx != 0:
            idx = 0
        frame = arr
    elif arr.ndim == 3:
        idx = max(0, min(idx, arr.shape[0] - 1))
        frame = arr[idx]
    else:
        raise ValueError(f"Unexpected TIFF shape at {path}: {arr.shape}")
    return np.asarray(frame)


def _seg_mask_2d(seg_path: str) -> np.ndarray:
    seg = tifffile.memmap(seg_path)
    seg_arr = np.asarray(seg)

    if seg_arr.ndim == 2:
        mask = seg_arr > 0
    elif seg_arr.ndim == 3:
        # legacy postprocessing output: (N_neurons, H, W)
        mask = np.max(seg_arr > 0, axis=0)
    else:
        raise ValueError(f"Unexpected segmentation TIFF shape: {seg_arr.shape}")

    return mask.astype(np.uint8)


def _overlay_mask(gray_frame: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray_frame, gray_frame, gray_frame], axis=-1).astype(np.uint8)
    m = mask2d.astype(bool)
    rgb[m] = np.array([255, 60, 60], dtype=np.uint8)
    return rgb


def _load_temporal_traces(mat_path: str) -> np.ndarray:
    mat = loadmat(mat_path)
    traces = np.asarray(mat.get("temporal_sig", np.array([])), dtype=np.float32)
    if traces.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if traces.ndim == 1:
        traces = traces[np.newaxis, :]
    elif traces.ndim > 2:
        traces = traces.reshape(traces.shape[0], -1)
    return traces


def _load_segmentation_channels(seg_path: str) -> np.ndarray:
    seg = np.asarray(tifffile.memmap(seg_path))
    if seg.ndim == 2:
        seg = seg[np.newaxis, ...]
    elif seg.ndim != 3:
        raise ValueError(f"Unexpected segmentation TIFF shape: {seg.shape}")
    return seg > 0


def get_trace_metadata(artifacts: dict) -> dict:
    traces = _load_temporal_traces(artifacts["temporal_traces_mat"])
    seg_channels = _load_segmentation_channels(artifacts["segmentation_tiff"])
    trace_count = int(traces.shape[0])
    seg_count = int(seg_channels.shape[0])
    usable = int(min(trace_count, seg_count))
    trace_warning = ""
    if trace_count != seg_count:
        trace_warning = (
            f"Trace/segmentation mismatch: traces={trace_count}, seg_channels={seg_count}. "
            f"Using first {usable} neurons."
        )

    return {
        "trace_count": usable,
        "trace_count_mat": trace_count,
        "segmentation_channel_count": seg_count,
        "trace_warning": trace_warning,
    }


def _build_trace_plot(trace: np.ndarray, frame_idx: int, neuron_idx: int) -> Figure:
    fig = Figure(figsize=(8, 2.6), dpi=100)
    ax = fig.subplots()

    x = np.arange(trace.shape[0], dtype=np.int32)
    ax.plot(x, trace, color="#1f77b4", linewidth=1.2)

    cursor = int(np.clip(frame_idx, 0, max(0, trace.shape[0] - 1)))
    ax.axvline(cursor, color="#d62728", linestyle="--", linewidth=1.0)
    ax.set_title(f"Temporal Trace (Neuron {neuron_idx + 1})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Signal")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return fig


def build_trace_study_assets(artifacts: dict, neuron_idx: int, frame_idx: int) -> dict:
    traces = _load_temporal_traces(artifacts["temporal_traces_mat"])
    seg_channels = _load_segmentation_channels(artifacts["segmentation_tiff"])

    trace_count = int(traces.shape[0])
    seg_count = int(seg_channels.shape[0])
    usable = int(min(trace_count, seg_count))
    warning = ""
    if trace_count != seg_count:
        warning = (
            f"Trace/segmentation mismatch: traces={trace_count}, seg_channels={seg_count}. "
            f"Using first {usable} neurons."
        )

    bg_frame_u8 = normalize_frame(_read_tiff_frame(artifacts["bg_rejected_tiff"], int(frame_idx)))

    if usable <= 0:
        empty = np.zeros((64,), dtype=np.float32)
        return {
            "selected_overlay": np.stack([bg_frame_u8, bg_frame_u8, bg_frame_u8], axis=-1),
            "trace_plot": _build_trace_plot(empty, frame_idx=0, neuron_idx=0),
            "trace_count": 0,
            "trace_warning": "No trace/segmentation neurons available.",
            "selected_neuron_idx": 0,
            "cursor_frame_idx": 0,
            "cursor_value": 0.0,
        }

    sel_idx = int(np.clip(neuron_idx, 0, usable - 1))
    mask2d = seg_channels[sel_idx].astype(np.uint8)
    overlay = _overlay_mask(bg_frame_u8, mask2d)

    trace = traces[sel_idx].astype(np.float32)
    if trace.size == 0:
        trace = np.zeros((1,), dtype=np.float32)
    cursor = int(np.clip(frame_idx, 0, max(0, trace.shape[0] - 1)))
    plot = _build_trace_plot(trace, frame_idx=cursor, neuron_idx=sel_idx)

    return {
        "selected_overlay": overlay,
        "trace_plot": plot,
        "trace_count": usable,
        "trace_warning": warning,
        "selected_neuron_idx": sel_idx,
        "cursor_frame_idx": cursor,
        "cursor_value": float(trace[cursor]),
    }


def build_preview_assets(artifacts: dict, frame_idx: int = 0) -> dict:
    input_frame = _read_tiff_frame(artifacts["input_tiff"], frame_idx)
    bg_frame = _read_tiff_frame(artifacts["background_tiff"], frame_idx)
    neuron_frame = _read_tiff_frame(artifacts["bg_rejected_tiff"], frame_idx)
    mask2d = _seg_mask_2d(artifacts["segmentation_tiff"])

    input_u8 = normalize_frame(input_frame)
    bg_u8 = normalize_frame(bg_frame)
    neuron_u8 = normalize_frame(neuron_frame)
    mask_u8 = (mask2d * 255).astype(np.uint8)
    overlay = _overlay_mask(neuron_u8, mask2d)

    input_arr = tifffile.memmap(artifacts["input_tiff"])
    frame_count = 1 if input_arr.ndim == 2 else int(input_arr.shape[0])

    return {
        "input": input_u8,
        "background": bg_u8,
        "bg_rejected": neuron_u8,
        "segmentation_mask": mask_u8,
        "segmentation_overlay": overlay,
        "frame_count": frame_count,
    }


def make_artifacts_zip(artifacts: dict, out_zip_path: str) -> str:
    run_dir = Path(artifacts["run_dir"]).resolve()
    legacy_output_dir = Path(artifacts["legacy_output_dir"]).resolve()
    input_dataset_tiff = Path(artifacts["input_dataset_tiff"]).resolve()
    manifest_path = Path(artifacts["manifest_json"]).resolve()

    files: list[Path] = []
    files.append(input_dataset_tiff)
    files.append(manifest_path)
    for p in sorted(legacy_output_dir.rglob("*")):
        if p.is_file():
            files.append(p.resolve())

    out_path = Path(out_zip_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[Path] = set()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            if not file_path.exists():
                continue
            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                arcname = str(resolved.relative_to(run_dir))
            except ValueError:
                arcname = resolved.name
            zf.write(str(resolved), arcname=arcname)

    return str(out_path)


def compute_parity_metrics(
    reference_artifacts: dict,
    candidate_artifacts: dict,
    traces_atol: float = 1e-6,
    traces_rtol: float = 1e-6,
) -> dict:
    """Compute deterministic parity metrics between two pipeline outputs."""
    metrics: dict[str, float | bool | tuple] = {}

    ref_bg = tifffile.imread(reference_artifacts["background_tiff"])
    cand_bg = tifffile.imread(candidate_artifacts["background_tiff"])
    ref_rmbg = tifffile.imread(reference_artifacts["bg_rejected_tiff"])
    cand_rmbg = tifffile.imread(candidate_artifacts["bg_rejected_tiff"])
    ref_seg = tifffile.imread(reference_artifacts["segmentation_tiff"])
    cand_seg = tifffile.imread(candidate_artifacts["segmentation_tiff"])

    metrics["background_shape_match"] = ref_bg.shape == cand_bg.shape
    metrics["bg_rejected_shape_match"] = ref_rmbg.shape == cand_rmbg.shape
    metrics["segmentation_shape_match"] = ref_seg.shape == cand_seg.shape

    bg_diff = np.abs(ref_bg.astype(np.float32) - cand_bg.astype(np.float32))
    rmbg_diff = np.abs(ref_rmbg.astype(np.float32) - cand_rmbg.astype(np.float32))

    metrics["background_max_abs_diff"] = float(bg_diff.max())
    metrics["background_mean_abs_diff"] = float(bg_diff.mean())
    metrics["bg_rejected_max_abs_diff"] = float(rmbg_diff.max())
    metrics["bg_rejected_mean_abs_diff"] = float(rmbg_diff.mean())

    ref_seg_bin = ref_seg > 0
    cand_seg_bin = cand_seg > 0
    metrics["segmentation_binary_equal"] = bool(np.array_equal(ref_seg_bin, cand_seg_bin))
    metrics["segmentation_mismatch_pixels"] = int(np.sum(ref_seg_bin != cand_seg_bin))

    ref_mat = loadmat(reference_artifacts["temporal_traces_mat"]) 
    cand_mat = loadmat(candidate_artifacts["temporal_traces_mat"]) 

    ref_traces = np.asarray(ref_mat.get("temporal_sig", np.array([])), dtype=np.float32)
    cand_traces = np.asarray(cand_mat.get("temporal_sig", np.array([])), dtype=np.float32)

    metrics["traces_shape_match"] = ref_traces.shape == cand_traces.shape
    if ref_traces.size == 0 and cand_traces.size == 0:
        traces_close = True
        traces_max_diff = 0.0
    elif ref_traces.shape != cand_traces.shape:
        traces_close = False
        traces_max_diff = float("inf")
    else:
        td = np.abs(ref_traces - cand_traces)
        traces_max_diff = float(td.max())
        traces_close = bool(np.allclose(ref_traces, cand_traces, rtol=traces_rtol, atol=traces_atol))

    metrics["traces_allclose"] = traces_close
    metrics["traces_max_abs_diff"] = traces_max_diff

    metrics["parity_pass"] = bool(
        metrics["background_shape_match"]
        and metrics["bg_rejected_shape_match"]
        and metrics["segmentation_shape_match"]
        and metrics["background_max_abs_diff"] == 0.0
        and metrics["bg_rejected_max_abs_diff"] == 0.0
        and metrics["segmentation_binary_equal"]
        and metrics["traces_allclose"]
    )

    return metrics
