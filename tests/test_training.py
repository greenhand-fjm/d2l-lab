"""端到端测试：在 CPU 上用合成数据跑几个 epoch，几秒内完成。"""

import json
import logging

import torch

from d2l_lab.config import Config
from d2l_lab.data import build_dataloaders
from d2l_lab.models import build_model
from d2l_lab.trainer import Trainer


def make_trainer(tmp_path, epochs: int, **overrides) -> Trainer:
    cfg = Config(
        name="t",
        output_dir=str(tmp_path),
        epochs=epochs,
        batch_size=10,
        lr=0.03,
        device="cpu",
        model_args={"in_features": 2, "out_features": 1},
        **overrides,
    )
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg.model, **cfg.model_args)
    return Trainer(cfg, model, train_loader, val_loader, logging.getLogger("test"))


def test_linear_regression_recovers_true_weights(tmp_path):
    trainer = make_trainer(tmp_path, epochs=3)
    before = trainer.evaluate()["val_loss"]
    after = trainer.fit()["val_loss"]
    assert after < 0.01 < before

    w = trainer.model.net.weight.detach().flatten()
    b = trainer.model.net.bias.detach()
    assert torch.allclose(w, torch.tensor([2.0, -3.4]), atol=0.05)
    assert torch.allclose(b, torch.tensor([4.2]), atol=0.05)


def test_resume_continues_from_last_checkpoint(tmp_path):
    make_trainer(tmp_path, epochs=2).fit()

    resumed = make_trainer(tmp_path, epochs=4)
    assert resumed.start_epoch == 2
    assert resumed.fit()["epoch"] == 4

    lines = (tmp_path / "t" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2, 3, 4]


def test_no_resume_starts_over(tmp_path):
    make_trainer(tmp_path, epochs=2).fit()
    fresh = make_trainer(tmp_path, epochs=2, resume=False)
    assert fresh.start_epoch == 0
