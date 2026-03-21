from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

SUPPORTED_TASKS = {
    "bg_crop",
    "bg_train",
    "bg_infer",
    "bg_post",
    "seg_crop",
    "seg_train",
    "seg_infer",
    "seg_post",
    "visualize",
    "infer_full",
}


def _cfg_to_dict(cfg_node: DictConfig) -> dict[str, Any]:
    data = OmegaConf.to_container(cfg_node, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Expected mapping config node")
    return data


def _normalize_cli_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(str(v) for v in value) + ")"
    return str(value)


def _build_cmd(script_name: str, params: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, script_name]
    for key, value in params.items():
        if value is None:
            continue

        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue

        cmd.extend([flag, _normalize_cli_value(value)])
    return cmd


def _run_cmd(cmd: list[str], cwd: Path, force_checkpoint_compat: bool) -> None:
    env = os.environ.copy()
    if force_checkpoint_compat:
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        stdout_tail = "\n".join(proc.stdout.splitlines()[-80:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-80:])
        raise RuntimeError(
            "Command failed\n"
            f"cwd: {cwd}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout (tail):\n{stdout_tail}\n"
            f"stderr (tail):\n{stderr_tail}"
        )


def _resolve_stage_dir(root_dir: Path, stage_name: str) -> Path:
    primary = root_dir / "codes" / stage_name
    legacy = root_dir / "legacy" / "codes" / stage_name

    if primary.exists():
        return primary
    if legacy.exists():
        return legacy

    raise FileNotFoundError(f"Unable to resolve stage directory for {stage_name}")


def _torch_load_compat(path: Path) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _convert_checkpoint_for_single_gpu(src_ckpt: Path, dst_ckpt: Path) -> None:
    ckpt = _torch_load_compat(src_ckpt)

    if not isinstance(ckpt, dict) or "net" not in ckpt or not isinstance(ckpt["net"], dict):
        dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
        dst_ckpt.write_bytes(src_ckpt.read_bytes())
        return

    state_dict = ckpt["net"]
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        ckpt["net"] = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(dst_ckpt))


def _prepare_ckpt_dir(
    ckpt_dir: str,
    stage_name: str,
    runtime_dir: Path,
    force_checkpoint_compat: bool,
) -> str:
    if not ckpt_dir:
        return ckpt_dir

    if not force_checkpoint_compat:
        return ckpt_dir

    src_ckpt = Path(ckpt_dir) / "checkpoint.pth"
    if not src_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {src_ckpt}")

    dst_ckpt = runtime_dir / "converted_ckpts" / stage_name / "checkpoint.pth"
    _convert_checkpoint_for_single_gpu(src_ckpt, dst_ckpt)
    return str(dst_ckpt.parent)


def _run_stage_script(
    root_dir: Path,
    stage_name: str,
    script_name: str,
    params: dict[str, Any],
    force_checkpoint_compat: bool,
) -> None:
    stage_dir = _resolve_stage_dir(root_dir, stage_name)
    cmd = _build_cmd(script_name, params)
    print(f"[ssfa] Running: {' '.join(cmd)}")
    _run_cmd(cmd=cmd, cwd=stage_dir, force_checkpoint_compat=force_checkpoint_compat)


def _run_bg_crop(cfg: DictConfig, root_dir: Path) -> str:
    params = _cfg_to_dict(cfg.bg.crop)
    _run_stage_script(root_dir, "BG_Rejection", "crop.py", params, bool(cfg.pipeline.force_checkpoint_compat))
    return str(params["output_dir"])


def _run_bg_train(cfg: DictConfig, root_dir: Path) -> str:
    params = _cfg_to_dict(cfg.bg.train)
    _run_stage_script(root_dir, "BG_Rejection", "main.py", params, bool(cfg.pipeline.force_checkpoint_compat))
    return str(params["output_dir"])


def _run_bg_infer(
    cfg: DictConfig,
    root_dir: Path,
    runtime_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> str:
    params = _cfg_to_dict(cfg.bg.infer)
    if overrides:
        params.update(overrides)

    params["ckpt_dir"] = _prepare_ckpt_dir(
        ckpt_dir=str(params["ckpt_dir"]),
        stage_name="bg",
        runtime_dir=runtime_dir,
        force_checkpoint_compat=bool(cfg.pipeline.force_checkpoint_compat),
    )

    _run_stage_script(root_dir, "BG_Rejection", "infer.py", params, bool(cfg.pipeline.force_checkpoint_compat))
    return str(params["output_dir"])


def _run_bg_post(
    cfg: DictConfig,
    root_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> str:
    params = _cfg_to_dict(cfg.bg.post)
    if overrides:
        params.update(overrides)

    _run_stage_script(
        root_dir,
        "BG_Rejection",
        "postprocessing_discard.py",
        params,
        bool(cfg.pipeline.force_checkpoint_compat),
    )
    return str(Path(params["dataset_dir"]) / str(params["dataset_name"]))


def _run_seg_crop(cfg: DictConfig, root_dir: Path) -> str:
    params = _cfg_to_dict(cfg.seg.crop)
    _run_stage_script(root_dir, "Segmentation", "crop.py", params, bool(cfg.pipeline.force_checkpoint_compat))
    return str(params["output_dir"])


def _run_seg_train(cfg: DictConfig, root_dir: Path) -> str:
    params = _cfg_to_dict(cfg.seg.train)
    _run_stage_script(root_dir, "Segmentation", "main.py", params, bool(cfg.pipeline.force_checkpoint_compat))
    return str(params["output_dir"])


def _run_seg_infer(
    cfg: DictConfig,
    root_dir: Path,
    runtime_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    params = _cfg_to_dict(cfg.seg.infer)
    if overrides:
        params.update(overrides)

    params["ckpt_dir"] = _prepare_ckpt_dir(
        ckpt_dir=str(params["ckpt_dir"]),
        stage_name="seg",
        runtime_dir=runtime_dir,
        force_checkpoint_compat=bool(cfg.pipeline.force_checkpoint_compat),
    )

    if str(params.get("training_mode", "joint")) == "sparse" and params.get("ckpt_dir2"):
        params["ckpt_dir2"] = _prepare_ckpt_dir(
            ckpt_dir=str(params["ckpt_dir2"]),
            stage_name="seg_sparse",
            runtime_dir=runtime_dir,
            force_checkpoint_compat=bool(cfg.pipeline.force_checkpoint_compat),
        )

    _run_stage_script(root_dir, "Segmentation", "infer.py", params, bool(cfg.pipeline.force_checkpoint_compat))

    dataset_name = str(params.get("dataset_name", ""))
    output_dir = Path(str(params["output_dir"]))
    return {
        "output_dir": str(output_dir),
        "latent_dir": str(output_dir / f"latent{dataset_name}"),
        "enc_feats_dir": str(output_dir / f"enc_feats{dataset_name}"),
    }


def _run_seg_post(
    cfg: DictConfig,
    root_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    params = _cfg_to_dict(cfg.seg.post)
    if overrides:
        params.update(overrides)

    _run_stage_script(root_dir, "Segmentation", "postprocessing.py", params, bool(cfg.pipeline.force_checkpoint_compat))

    dataset_name = str(params.get("dataset_name", ""))
    output_dir = Path(str(params["output_dir"]))
    return {
        "output_dir": str(output_dir),
        "seg_dir": str(output_dir / f"seg{dataset_name}"),
        "sig_dir": str(output_dir / f"sig{dataset_name}"),
    }


def _run_visualize(cfg: DictConfig, root_dir: Path) -> str:
    params = _cfg_to_dict(cfg.pipeline.visualize)
    _run_stage_script(root_dir, "Visualization", "gen_footprints.py", params, bool(cfg.pipeline.force_checkpoint_compat))
    return str(params["output_dir"])


def _run_infer_full(cfg: DictConfig, root_dir: Path, runtime_dir: Path) -> dict[str, str]:
    output_root = Path(str(cfg.pipeline.output_dir))
    bg_stage_output = output_root / "bg_stage"
    seg_stage_output = output_root / "seg_stage"
    final_output = output_root / "final_output"

    bg_output_dir = _run_bg_infer(
        cfg,
        root_dir,
        runtime_dir,
        overrides={
            "dataset_dir": str(cfg.pipeline.input_dir),
            "output_dir": str(bg_stage_output),
            "ckpt_dir": str(cfg.pipeline.bg_ckpt_dir),
            "device": str(cfg.pipeline.device),
            "gpu_ids": str(cfg.pipeline.gpu_ids),
        },
    )

    rmbg_dir = _run_bg_post(
        cfg,
        root_dir,
        overrides={
            "dataset_dir": bg_output_dir,
            "dataset_name": str(cfg.bg.post.dataset_name),
        },
    )

    seg_outputs = _run_seg_infer(
        cfg,
        root_dir,
        runtime_dir,
        overrides={
            "dataset_dir": rmbg_dir,
            "output_dir": str(seg_stage_output),
            "ckpt_dir": str(cfg.pipeline.seg_ckpt_dir),
            "device": str(cfg.pipeline.device),
            "gpu_ids": str(cfg.pipeline.gpu_ids),
        },
    )

    post_outputs = _run_seg_post(
        cfg,
        root_dir,
        overrides={
            "rmbg_dir": rmbg_dir,
            "latent_dir": seg_outputs["latent_dir"],
            "output_dir": str(final_output),
        },
    )

    return {
        "bg_stage": bg_output_dir,
        "rmbg_dir": rmbg_dir,
        "seg_stage": seg_outputs["output_dir"],
        "latent_dir": seg_outputs["latent_dir"],
        "final_output": post_outputs["output_dir"],
        "seg_dir": post_outputs["seg_dir"],
        "sig_dir": post_outputs["sig_dir"],
    }


def _dispatch_task(cfg: DictConfig, root_dir: Path, runtime_dir: Path) -> dict[str, str]:
    task = str(cfg.task)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Choose one of: {sorted(SUPPORTED_TASKS)}")

    if task == "bg_crop":
        return {"output_dir": _run_bg_crop(cfg, root_dir)}
    if task == "bg_train":
        return {"output_dir": _run_bg_train(cfg, root_dir)}
    if task == "bg_infer":
        return {"output_dir": _run_bg_infer(cfg, root_dir, runtime_dir)}
    if task == "bg_post":
        return {"output_dir": _run_bg_post(cfg, root_dir)}
    if task == "seg_crop":
        return {"output_dir": _run_seg_crop(cfg, root_dir)}
    if task == "seg_train":
        return {"output_dir": _run_seg_train(cfg, root_dir)}
    if task == "seg_infer":
        return _run_seg_infer(cfg, root_dir, runtime_dir)
    if task == "seg_post":
        return _run_seg_post(cfg, root_dir)
    if task == "visualize":
        return {"output_dir": _run_visualize(cfg, root_dir)}
    if task == "infer_full":
        return _run_infer_full(cfg, root_dir, runtime_dir)

    raise AssertionError("Unreachable task dispatch")


@hydra.main(version_base="1.3", config_path="../configs", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    print("[ssfa] Starting pipeline")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    root_dir = Path(str(cfg.paths.root_dir))
    runtime_dir = Path(HydraConfig.get().runtime.output_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    outputs = _dispatch_task(cfg, root_dir, runtime_dir)
    print("[ssfa] Task complete")
    for key, value in outputs.items():
        print(f"[ssfa] {key}: {value}")


if __name__ == "__main__":
    main()
