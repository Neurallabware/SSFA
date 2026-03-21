from __future__ import annotations

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from webapp.app import launch_webui


@hydra.main(version_base="1.3", config_path="../configs", config_name="webui")
def main(cfg: DictConfig) -> None:
    print("[ssfa] Starting web UI")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    launch_webui(
        port=int(cfg.webui.port),
        share=bool(cfg.webui.share),
        device=int(cfg.webui.device),
        device_type=str(cfg.webui.device_type),
        work_dir=str(cfg.webui.work_dir),
        max_workers=int(cfg.webui.max_workers),
        bg_ckpt_dir=str(cfg.webui.bg_ckpt_dir),
        seg_ckpt_dir=str(cfg.webui.seg_ckpt_dir),
        server_name=str(cfg.webui.server_name),
    )


if __name__ == "__main__":
    main()
