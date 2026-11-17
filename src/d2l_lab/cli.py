"""命令行入口。

python -m d2l_lab.cli --config configs/linreg.yaml
python -m d2l_lab.cli --config configs/linreg.yaml --set lr=0.1 --set epochs=10
d2l-train --config configs/mlp_fashion.yaml --no-resume
"""

import argparse
import sys

from .config import Config
from .data import build_dataloaders
from .models import build_model
from .trainer import Trainer
from .utils import seed_everything, setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a D2L model from a YAML config.")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value; repeat for several",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="ignore old checkpoints and start from scratch"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = list(args.overrides)
    if args.no_resume:
        overrides.append("resume=false")
    cfg = Config.from_yaml(args.config, overrides)

    logger = setup_logger(cfg.run_dir)
    cfg.save(cfg.run_dir / "config.yaml")
    seed_everything(cfg.seed)
    logger.info("run dir: %s", cfg.run_dir.resolve())

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg.model, **cfg.model_args)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model: %s (%s parameters)", cfg.model, f"{n_params:,}")

    trainer = Trainer(cfg, model, train_loader, val_loader, logger)
    logger.info("device: %s%s", trainer.device, " + bf16 autocast" if trainer.use_amp else "")
    final = trainer.fit()
    logger.info("finished: %s", final)
    return 0


# Windows 上 DataLoader 用多进程（num_workers > 0）时，子进程会重新导入主模块。
# 没有这个保护，子进程会再次启动训练，无限套娃后报错。
if __name__ == "__main__":
    sys.exit(main())
