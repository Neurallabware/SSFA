"""Deterministic parity check between local baseline and webapp backend pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from .inference import compute_parity_metrics, run_legacy_full_pipeline
    from .pipeline_legacy import DEFAULT_BG_CKPT_DIR, DEFAULT_SEG_CKPT_DIR
except ImportError:
    from inference import compute_parity_metrics, run_legacy_full_pipeline
    from pipeline_legacy import DEFAULT_BG_CKPT_DIR, DEFAULT_SEG_CKPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSFA webapp parity checker for local MP4/TIFF pipeline")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "local" / "supplementary_video_8.mp4"),
        help="Input MP4/TIFF path",
    )
    parser.add_argument(
        "--workspace",
        default=str(Path(__file__).resolve().parent / "parity_runs"),
        help="Workspace directory for parity artifacts",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--batch-size", type=int, default=1, help="Legacy infer batch size")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quicker checks")
    parser.add_argument("--traces-atol", type=float, default=1e-6)
    parser.add_argument("--traces-rtol", type=float, default=1e-6)
    parser.add_argument("--bg-ckpt-dir", default=str(DEFAULT_BG_CKPT_DIR))
    parser.add_argument("--seg-ckpt-dir", default=str(DEFAULT_SEG_CKPT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    baseline_dir = workspace / "baseline_local"
    webapp_dir = workspace / "webapp_backend"

    print("[1/3] Running local baseline pipeline...")
    baseline_artifacts = run_legacy_full_pipeline(
        input_path=str(input_path),
        run_dir=str(baseline_dir),
        device_idx=args.device,
        batch_size=args.batch_size,
        bg_ckpt_dir=args.bg_ckpt_dir,
        seg_ckpt_dir=args.seg_ckpt_dir,
        max_frames=args.max_frames,
    )

    print("[2/3] Running webapp backend pipeline...")
    webapp_artifacts = run_legacy_full_pipeline(
        input_path=str(input_path),
        run_dir=str(webapp_dir),
        device_idx=args.device,
        batch_size=args.batch_size,
        bg_ckpt_dir=args.bg_ckpt_dir,
        seg_ckpt_dir=args.seg_ckpt_dir,
        max_frames=args.max_frames,
    )

    print("[3/3] Computing parity metrics...")
    metrics = compute_parity_metrics(
        baseline_artifacts,
        webapp_artifacts,
        traces_atol=args.traces_atol,
        traces_rtol=args.traces_rtol,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))

    return 0 if metrics.get("parity_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
