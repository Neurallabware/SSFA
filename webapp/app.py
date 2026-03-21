"""SSFA webapp with async queued full-pipeline parity for MP4/TIFF."""

from __future__ import annotations

import argparse
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gradio as gr
import tifffile

try:
    from .inference import (
        build_preview_assets,
        build_trace_study_assets,
        make_artifacts_zip,
        run_legacy_full_pipeline,
    )
    from .pipeline_legacy import DEFAULT_BG_CKPT_DIR, DEFAULT_SEG_CKPT_DIR
except ImportError:
    from inference import (
        build_preview_assets,
        build_trace_study_assets,
        make_artifacts_zip,
        run_legacy_full_pipeline,
    )
    from pipeline_legacy import DEFAULT_BG_CKPT_DIR, DEFAULT_SEG_CKPT_DIR


@dataclass
class JobRecord:
    job_id: str
    input_path: str
    created_at: float
    run_dir: str
    status: str = "queued"  # queued|running|failed|done
    progress: float = 0.0
    stage: str = "queued"
    error: str = ""
    artifacts: dict = field(default_factory=dict)


class JobManager:
    def __init__(
        self,
        work_dir: str,
        bg_ckpt_dir: str,
        seg_ckpt_dir: str,
        device_idx: int,
        device_type: str | None = None,
        max_workers: int = 1,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.bg_ckpt_dir = bg_ckpt_dir
        self.seg_ckpt_dir = seg_ckpt_dir
        self.device_idx = device_idx
        self.device_type = device_type

        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="webapp-worker")
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(self, input_file: str, batch_size: int, max_frames: int | None) -> str:
        job_id = uuid.uuid4().hex[:12]
        run_dir = self.work_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)

        src = Path(input_file)
        copied = run_dir / f"upload{src.suffix.lower()}"
        shutil.copy2(src, copied)

        rec = JobRecord(
            job_id=job_id,
            input_path=str(copied),
            created_at=time.time(),
            run_dir=str(run_dir),
        )
        with self._lock:
            self._jobs[job_id] = rec

        self._pool.submit(self._run, job_id, int(batch_size), max_frames)
        return job_id

    def _update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            for k, v in kwargs.items():
                setattr(rec, k, v)

    def _run(self, job_id: str, batch_size: int, max_frames: int | None) -> None:
        try:
            self._update(job_id, status="running", stage="starting", progress=0.01)

            def cb(frac: float, msg: str):
                self._update(job_id, status="running", progress=float(frac), stage=str(msg))

            rec = self.get(job_id)
            artifacts = run_legacy_full_pipeline(
                input_path=rec["input_path"],
                run_dir=rec["run_dir"],
                device_idx=self.device_idx,
                device_type=self.device_type,
                batch_size=batch_size,
                bg_ckpt_dir=self.bg_ckpt_dir,
                seg_ckpt_dir=self.seg_ckpt_dir,
                max_frames=max_frames,
                progress_callback=cb,
            )

            zip_path = str(Path(rec["run_dir"]) / "artifacts" / "pipeline_outputs.zip")
            make_artifacts_zip(artifacts, zip_path)
            artifacts["full_pipeline_zip"] = zip_path
            artifacts["artifacts_zip"] = zip_path

            self._update(job_id, status="done", progress=1.0, stage="done", artifacts=artifacts)
        except Exception as exc:
            self._update(job_id, status="failed", stage="failed", error=str(exc), progress=1.0)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return {
                "job_id": rec.job_id,
                "status": rec.status,
                "progress": rec.progress,
                "stage": rec.stage,
                "error": rec.error,
                "artifacts": dict(rec.artifacts),
                "input_path": rec.input_path,
                "run_dir": rec.run_dir,
                "created_at": rec.created_at,
            }


JOB_MANAGER: JobManager | None = None


def describe_input(file_path: str) -> str:
    p = Path(file_path)
    ext = p.suffix.lower()

    if ext in {".tif", ".tiff"}:
        arr = tifffile.memmap(str(p))
        if arr.ndim == 2:
            shape = (1, arr.shape[0], arr.shape[1])
        elif arr.ndim == 3:
            shape = arr.shape
        else:
            shape = arr.shape
        size_mb = p.stat().st_size / (1024 * 1024)
        return f"Input: TIFF | shape={shape} | dtype={arr.dtype} | size={size_mb:.1f} MB"

    if ext == ".mp4":
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            return "Input: MP4 | failed to open"
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        size_mb = p.stat().st_size / (1024 * 1024)
        return (
            f"Input: MP4 | frames={frames} | fps={fps:.3f} | "
            f"size={width}x{height} | file={size_mb:.1f} MB"
        )

    return f"Unsupported file type: {ext}"


def on_upload(file):
    if file is None:
        return "No file uploaded. Allowed: mp4, tif, tiff"
    return describe_input(file.name)


def _download_updates_disabled():
    return (
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
    )


def _download_updates_ready(artifacts: dict):
    return (
        gr.update(value=artifacts["full_pipeline_zip"], interactive=True),
        gr.update(value=artifacts["background_tiff"], interactive=True),
        gr.update(value=artifacts["bg_rejected_tiff"], interactive=True),
        gr.update(value=artifacts["segmentation_tiff"], interactive=True),
        gr.update(value=artifacts["temporal_traces_mat"], interactive=True),
        gr.update(value=artifacts["manifest_json"], interactive=True),
    )


def _format_nav_status(frame_idx: int, frame_count: int, trace_assets: dict) -> str:
    if frame_count <= 0:
        return "Frame=N/A | Neuron=N/A"

    selected = int(trace_assets.get("selected_neuron_idx", 0))
    trace_count = int(trace_assets.get("trace_count", 0))
    neuron_text = f"{selected + 1}/{trace_count}" if trace_count > 0 else "N/A"
    msg = f"Frame={int(frame_idx) + 1}/{int(frame_count)} | Neuron={neuron_text}"

    warning = str(trace_assets.get("trace_warning", "")).strip()
    if warning:
        msg = f"{msg} | {warning}"
    return msg


def _empty_outputs(msg: str):
    return (
        msg,
        "",
        gr.update(active=False),
        gr.update(minimum=0, maximum=0, value=0, visible=False),
        gr.update(minimum=0, maximum=0, value=0, step=1, visible=False, interactive=False),
        gr.update(value="", visible=False),
        None,
        None,
        None,
        None,
        None,
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        *_download_updates_disabled(),
        {},
    )


def submit_job(file, batch_size, max_frames):
    if file is None:
        return _empty_outputs("Please upload an MP4 or TIFF file first.")

    ext = Path(file.name).suffix.lower()
    if ext not in {".mp4", ".tif", ".tiff"}:
        return _empty_outputs(f"Unsupported file type: {ext}. Allowed: mp4, tif, tiff")

    assert JOB_MANAGER is not None

    mf_raw = 0 if max_frames is None else int(max_frames)
    mf = mf_raw if mf_raw > 0 else None
    job_id = JOB_MANAGER.submit(file.name, batch_size=int(batch_size), max_frames=mf)

    status = f"Job {job_id} submitted."
    verify = ""

    return (
        status,
        verify,
        gr.update(active=True),
        gr.update(minimum=0, maximum=0, value=0, visible=True, interactive=False),
        gr.update(minimum=0, maximum=0, value=0, step=1, visible=False, interactive=False),
        gr.update(value="Waiting for results...", visible=True),
        None,
        None,
        None,
        None,
        None,
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        *_download_updates_disabled(),
        {"job_id": job_id, "selected_neuron": 0},
    )


def _format_trace_info(trace_assets: dict) -> str:
    count = int(trace_assets.get("trace_count", 0))
    if count <= 0:
        return str(trace_assets.get("trace_warning", "No traces available."))

    idx = int(trace_assets["selected_neuron_idx"]) + 1
    cursor = int(trace_assets["cursor_frame_idx"])
    value = float(trace_assets["cursor_value"])
    info = f"Neurons={count} | selected={idx}/{count} | frame={cursor} | value={value:.6f}"
    warning = str(trace_assets.get("trace_warning", "")).strip()
    if warning:
        info = f"{info} | {warning}"
    return info


def poll_job(state):
    if not state or "job_id" not in state:
        return _empty_outputs("No active job.")

    job_id = state["job_id"]
    assert JOB_MANAGER is not None
    rec = JOB_MANAGER.get(job_id)

    if rec is None:
        return _empty_outputs(f"Job {job_id} not found.")

    status = rec["status"]
    progress = rec["progress"]
    stage = rec["stage"]
    status_text = f"Job {job_id} | status={status} | progress={progress:.0%} | stage={stage}"

    if status in {"queued", "running"}:
        return (
            status_text,
            "",
            gr.update(active=True),
            gr.update(minimum=0, maximum=0, value=0, visible=True, interactive=False),
            gr.update(minimum=0, maximum=0, value=0, step=1, visible=False, interactive=False),
            gr.update(value="Processing... navigation activates when job is done.", visible=True),
            None,
            None,
            None,
            None,
            None,
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            *_download_updates_disabled(),
            state,
        )

    if status == "failed":
        return (
            status_text,
            f"Error: {rec['error']}",
            gr.update(active=False),
            gr.update(minimum=0, maximum=0, value=0, visible=True, interactive=False),
            gr.update(minimum=0, maximum=0, value=0, step=1, visible=False, interactive=False),
            gr.update(value="Run failed. Fix error and submit again.", visible=True),
            None,
            None,
            None,
            None,
            None,
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            *_download_updates_disabled(),
            state,
        )

    # done
    artifacts = rec["artifacts"]
    preview = build_preview_assets(artifacts, frame_idx=0)
    trace_assets = build_trace_study_assets(artifacts, neuron_idx=0, frame_idx=0)
    verify = (
        f"Reconstruction check: max_diff={float(artifacts['reconstruction_max_diff']):.2f}, "
        f"mean_diff={float(artifacts['reconstruction_mean_diff']):.2f}"
    )
    if artifacts.get("trace_warning"):
        verify = f"{verify} | {artifacts['trace_warning']}"

    frame_count = int(preview["frame_count"])
    slider_update = gr.update(
        minimum=0,
        maximum=max(0, frame_count - 1),
        value=0,
        visible=True,
        interactive=True,
    )
    if int(trace_assets.get("trace_count", 0)) > 0:
        neuron_slider = gr.update(
            minimum=0,
            maximum=max(0, int(trace_assets["trace_count"]) - 1),
            value=int(trace_assets["selected_neuron_idx"]),
            step=1,
            visible=True,
            interactive=True,
        )
    else:
        neuron_slider = gr.update(minimum=0, maximum=0, value=0, step=1, visible=False, interactive=False)

    done_state = {
        "job_id": job_id,
        "artifacts": artifacts,
        "selected_neuron": int(trace_assets.get("selected_neuron_idx", 0)),
        "frame_count": frame_count,
    }

    return (
        status_text,
        verify,
        gr.update(active=False),
        slider_update,
        neuron_slider,
        gr.update(value=_format_nav_status(0, frame_count, trace_assets), visible=True),
        preview["input"],
        preview["background"],
        preview["bg_rejected"],
        preview["segmentation_mask"],
        trace_assets["selected_overlay"],
        gr.update(value=trace_assets["trace_plot"], visible=True),
        gr.update(value=_format_trace_info(trace_assets), visible=True),
        *_download_updates_ready(artifacts),
        done_state,
    )


def update_frame(frame_idx, state):
    if not state or "artifacts" not in state:
        return (
            None,
            None,
            None,
            None,
            None,
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            state,
        )

    artifacts = state["artifacts"]
    idx = int(frame_idx)
    preview = build_preview_assets(artifacts, frame_idx=idx)
    sel = int(state.get("selected_neuron", 0))
    trace_assets = build_trace_study_assets(artifacts, neuron_idx=sel, frame_idx=idx)
    frame_count = int(state.get("frame_count", preview["frame_count"]))
    next_state = dict(state)
    next_state["selected_neuron"] = int(trace_assets["selected_neuron_idx"])
    next_state["frame_count"] = frame_count

    return (
        preview["input"],
        preview["background"],
        preview["bg_rejected"],
        preview["segmentation_mask"],
        trace_assets["selected_overlay"],
        gr.update(value=trace_assets["trace_plot"], visible=True),
        gr.update(value=_format_trace_info(trace_assets), visible=True),
        gr.update(value=_format_nav_status(idx, frame_count, trace_assets), visible=True),
        next_state,
    )


def update_neuron(neuron_idx, frame_idx, state):
    if not state or "artifacts" not in state:
        return (
            None,
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            state,
        )

    artifacts = state["artifacts"]
    idx = int(frame_idx)
    trace_assets = build_trace_study_assets(artifacts, neuron_idx=int(neuron_idx), frame_idx=idx)
    frame_count = int(state.get("frame_count", idx + 1))
    next_state = dict(state)
    next_state["selected_neuron"] = int(trace_assets["selected_neuron_idx"])
    next_state["frame_count"] = frame_count
    return (
        trace_assets["selected_overlay"],
        gr.update(value=trace_assets["trace_plot"], visible=True),
        gr.update(value=_format_trace_info(trace_assets), visible=True),
        gr.update(value=_format_nav_status(idx, frame_count, trace_assets), visible=True),
        next_state,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SSFA — MP4/TIFF Full Pipeline") as demo:
        gr.Markdown(
            "# SSFA — Full Pipeline (MP4/TIFF)\n"
            "Upload MP4 or TIFF, run background rejection + segmentation, then download all artifacts."
        )

        state = gr.State({})

        with gr.Row():
            file_input = gr.File(label="Upload Input", file_types=[".mp4", ".tif", ".tiff"])
            run_btn = gr.Button("Submit Job", variant="primary", size="lg")
        with gr.Row():
            dl_zip = gr.DownloadButton(
                label="Download Full ZIP",
                variant="primary",
                size="sm",
                interactive=False,
            )
            dl_bg = gr.DownloadButton(label="Background", size="sm", interactive=False)
            dl_rmbg = gr.DownloadButton(label="BG-Rejected", size="sm", interactive=False)
            dl_seg = gr.DownloadButton(label="Segmentation", size="sm", interactive=False)
            dl_sig = gr.DownloadButton(label="Traces (.mat)", size="sm", interactive=False)
            dl_manifest = gr.DownloadButton(label="Manifest (.json)", size="sm", interactive=False)

        info_text = gr.Textbox(label="Input Info", interactive=False)
        status_text = gr.Textbox(label="Job Status", interactive=False)
        verify_text = gr.Textbox(label="Verification", interactive=False)

        with gr.Accordion("Advanced Settings", open=False):
            batch_size = gr.Slider(minimum=1, maximum=8, value=1, step=1, label="Pipeline infer batch size")
            max_frames = gr.Number(value=0, precision=0, label="Max frames (0 = full video)")

        with gr.Row():
            frame_slider = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="Frame",
                visible=False,
                interactive=False,
                scale=4,
            )
            neuron_slider = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="Neuron Index",
                visible=False,
                interactive=False,
                scale=3,
            )
            nav_text = gr.Textbox(label="Navigation", interactive=False, visible=False, scale=3)

        with gr.Row():
            img_input = gr.Image(label="Input", type="numpy")
            img_bg = gr.Image(label="Background", type="numpy")
            img_rmbg = gr.Image(label="BG-Rejected", type="numpy")
        with gr.Row():
            img_seg = gr.Image(label="Segmentation Mask", type="numpy")
            img_overlay = gr.Image(label="Selected Neuron Overlay", type="numpy")
        trace_plot = gr.Plot(label="Temporal Trace", visible=False)
        trace_info = gr.Textbox(label="Trace Info", interactive=False, visible=False)

        poll_timer = gr.Timer(value=1.0, active=False)

        file_input.change(fn=on_upload, inputs=[file_input], outputs=[info_text])

        run_btn.click(
            fn=submit_job,
            inputs=[file_input, batch_size, max_frames],
            outputs=[
                status_text,
                verify_text,
                poll_timer,
                frame_slider,
                neuron_slider,
                nav_text,
                img_input,
                img_bg,
                img_rmbg,
                img_seg,
                img_overlay,
                trace_plot,
                trace_info,
                dl_zip,
                dl_bg,
                dl_rmbg,
                dl_seg,
                dl_sig,
                dl_manifest,
                state,
            ],
        )

        poll_timer.tick(
            fn=poll_job,
            inputs=[state],
            outputs=[
                status_text,
                verify_text,
                poll_timer,
                frame_slider,
                neuron_slider,
                nav_text,
                img_input,
                img_bg,
                img_rmbg,
                img_seg,
                img_overlay,
                trace_plot,
                trace_info,
                dl_zip,
                dl_bg,
                dl_rmbg,
                dl_seg,
                dl_sig,
                dl_manifest,
                state,
            ],
        )

        frame_slider.change(
            fn=update_frame,
            inputs=[frame_slider, state],
            outputs=[img_input, img_bg, img_rmbg, img_seg, img_overlay, trace_plot, trace_info, nav_text, state],
        )

        neuron_slider.change(
            fn=update_neuron,
            inputs=[neuron_slider, frame_slider, state],
            outputs=[img_overlay, trace_plot, trace_info, nav_text, state],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSFA webapp")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--device-type", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parent / "jobs"))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--bg-ckpt-dir", default=str(DEFAULT_BG_CKPT_DIR))
    parser.add_argument("--seg-ckpt-dir", default=str(DEFAULT_SEG_CKPT_DIR))
    parser.add_argument("--server-name", default="0.0.0.0")
    return parser.parse_args()


def launch_webui(
    port: int = 7860,
    share: bool = False,
    device: int = 0,
    device_type: str = "auto",
    work_dir: str = str(Path(__file__).resolve().parent / "jobs"),
    max_workers: int = 1,
    bg_ckpt_dir: str = str(DEFAULT_BG_CKPT_DIR),
    seg_ckpt_dir: str = str(DEFAULT_SEG_CKPT_DIR),
    server_name: str = "0.0.0.0",
) -> None:
    global JOB_MANAGER
    JOB_MANAGER = JobManager(
        work_dir=work_dir,
        bg_ckpt_dir=bg_ckpt_dir,
        seg_ckpt_dir=seg_ckpt_dir,
        device_idx=device,
        device_type=None if device_type == "auto" else device_type,
        max_workers=max_workers,
    )

    app = build_app()
    app.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        max_file_size="4gb",
        theme=gr.themes.Soft(),
    )


def main() -> None:
    args = parse_args()
    launch_webui(
        port=args.port,
        share=args.share,
        device=args.device,
        device_type=args.device_type,
        work_dir=args.work_dir,
        max_workers=args.max_workers,
        bg_ckpt_dir=args.bg_ckpt_dir,
        seg_ckpt_dir=args.seg_ckpt_dir,
        server_name=args.server_name,
    )


if __name__ == "__main__":
    main()
