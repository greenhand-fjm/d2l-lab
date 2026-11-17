"""训练循环。所有模型共用这一份代码：前向、反向、评估、记录、保存断点。

工程要点
- 一份训练循环服务所有模型，而不是每章复制粘贴一遍。
- 断点续训：每个 epoch 结束保存 last.pt（模型 + 优化器 + 进度），
  笔记本休眠、程序崩溃后重新运行同一条命令即可接着训练。
- 原子写入：先写临时文件再改名，写到一半断电也不会留下损坏的检查点。
- 指标落盘：每个 epoch 一行 JSON 写进 metrics.jsonl，方便之后画图对比实验。
"""

import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .utils import get_device


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger("d2l_lab")
        self.device = get_device(cfg.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self._build_optimizer()
        self.loss_fn = nn.MSELoss() if cfg.task == "regression" else nn.CrossEntropyLoss()
        self.use_amp = cfg.amp and self.device.type == "cuda"

        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0
        self.best_val_loss = math.inf
        last = self.run_dir / "last.pt"
        if cfg.resume and last.exists():
            self.load_checkpoint(last)
        else:
            # 从头训练时清掉旧的指标记录，避免新旧两次实验混在一个文件里
            (self.run_dir / "metrics.jsonl").unlink(missing_ok=True)

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        lr, wd = self.cfg.lr, self.cfg.weight_decay
        name = self.cfg.optimizer.lower()
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd)
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        raise ValueError(f"unknown optimizer {self.cfg.optimizer!r}; use sgd, adam or adamw")

    def _autocast(self):
        # bf16 混合精度：矩阵乘法用 bf16 算，更快、更省显存；50 系显卡原生支持
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp)

    # --------------------------------------------------------------- training
    def fit(self) -> dict[str, float]:
        if self.start_epoch >= self.cfg.epochs:
            self.logger.info(
                "already trained %d/%d epochs, nothing to do (raise epochs or pass --no-resume)",
                self.start_epoch,
                self.cfg.epochs,
            )
        metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, self.cfg.epochs):
            start = time.perf_counter()
            train_loss = self.train_one_epoch(epoch)
            metrics = {"epoch": epoch + 1, "train_loss": train_loss, **self.evaluate()}
            metrics["seconds"] = round(time.perf_counter() - start, 2)
            self._log_epoch(metrics)

            is_best = metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = metrics["val_loss"]
            self.save_checkpoint(self.run_dir / "last.pt", epoch + 1)
            if is_best:
                self.save_checkpoint(self.run_dir / "best.pt", epoch + 1)
        return metrics

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total, count = 0.0, 0
        for step, (X, y) in enumerate(self.train_loader):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                loss = self.loss_fn(self.model(X), y)
            if not torch.isfinite(loss):
                # 学习率过大时 loss 会变成 inf/nan，立刻停下比白跑几个小时好
                raise FloatingPointError(
                    f"loss is {loss.item()} at epoch {epoch + 1}, step {step}; try a smaller lr"
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total += loss.item() * X.shape[0]
            count += X.shape[0]
            if self.cfg.log_every and step % self.cfg.log_every == 0:
                self.logger.debug("epoch %d step %d loss %.4f", epoch + 1, step, loss.item())
        return total / count

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()  # 关掉 dropout 等只在训练时生效的层
        total_loss, correct, count = 0.0, 0, 0
        for X, y in self.val_loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                out = self.model(X)
                loss = self.loss_fn(out, y)
            total_loss += loss.item() * X.shape[0]
            count += X.shape[0]
            if self.cfg.task == "classification":
                correct += (out.argmax(dim=1) == y).sum().item()
        result = {"val_loss": total_loss / count}
        if self.cfg.task == "classification":
            result["val_acc"] = correct / count
        return result

    # ---------------------------------------------------------- bookkeeping
    def _log_epoch(self, metrics: dict[str, float]) -> None:
        text = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
        )
        self.logger.info(text)
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg),
        }
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)  # 原子替换：要么是旧文件，要么是完整的新文件

    def load_checkpoint(self, path: Path) -> None:
        # weights_only=True：只加载张量和基本类型，不执行文件里的任意代码
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.start_epoch = state["epoch"]
        self.best_val_loss = state["best_val_loss"]
        self.logger.info("resumed from %s (finished epoch %d)", path, self.start_epoch)
