"""通用工具：随机种子、设备选择、日志。"""

import logging
import random
import sys
from pathlib import Path

import torch


def seed_everything(seed: int) -> None:
    """固定所有随机源，同一份配置跑两次能得到相同的结果（可复现性）。"""
    random.seed(seed)
    torch.manual_seed(seed)  # 同时设置 CPU 和所有 GPU 的种子
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def get_device(name: str = "auto") -> torch.device:
    """auto: 有 GPU 用 GPU，没有就用 CPU。显式要求 cuda 却没有 GPU 时直接报错。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for CUDA but torch.cuda.is_available() is False")
    return device


def setup_logger(run_dir: Path | None = None, name: str = "d2l_lab") -> logging.Logger:
    """同时输出到终端和 run 目录下的 train.log。

    用 logging 而不是 print：每行带时间、能按级别过滤、能同时写进文件，
    训练结束后还能回头查。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):  # 重复调用时不叠加 handler，避免每行打印两次
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
