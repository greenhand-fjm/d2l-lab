"""数据集注册表。

工程要点：新增一个数据集 = 写一个函数 + 加一行 @register_dataset("名字")。
训练代码只认名字，不需要任何改动（开闭原则：对扩展开放，对修改关闭）。
"""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from .config import Config

DatasetPair = tuple[Dataset, Dataset]
_DATASETS: dict[str, Callable[[Config], DatasetPair]] = {}


def register_dataset(name: str) -> Callable:
    def decorator(fn: Callable[[Config], DatasetPair]) -> Callable[[Config], DatasetPair]:
        if name in _DATASETS:
            raise ValueError(f"dataset {name!r} registered twice")
        _DATASETS[name] = fn
        return fn

    return decorator


def synthetic_linear(
    n: int, w: tuple[float, ...], b: float, noise: float, seed: int
) -> TensorDataset:
    """D2L 3.3：按 y = Xw + b + 噪声 生成数据。用独立的 Generator，不受全局随机状态影响。"""
    gen = torch.Generator().manual_seed(seed)
    true_w = torch.tensor(w, dtype=torch.float32)
    X = torch.randn(n, len(true_w), generator=gen)
    y = X @ true_w + b + noise * torch.randn(n, generator=gen)
    return TensorDataset(X, y.reshape(-1, 1))


@register_dataset("synthetic_linear")
def _synthetic_linear(cfg: Config) -> DatasetPair:
    full = synthetic_linear(n=1000, w=(2.0, -3.4), b=4.2, noise=0.01, seed=cfg.seed)
    train, val = random_split(full, [900, 100], generator=torch.Generator().manual_seed(cfg.seed))
    return train, val


@register_dataset("fashion_mnist")
def _fashion_mnist(cfg: Config) -> DatasetPair:
    # 延迟导入：只跑合成数据时不必加载 torchvision
    from torchvision import datasets, transforms

    to_tensor = transforms.ToTensor()
    train = datasets.FashionMNIST(cfg.data_dir, train=True, download=True, transform=to_tensor)
    val = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=to_tensor)
    return train, val


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    if cfg.dataset not in _DATASETS:
        raise KeyError(f"unknown dataset {cfg.dataset!r}; available: {sorted(_DATASETS)}")
    train_set, val_set = _DATASETS[cfg.dataset](cfg)

    common = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        # pin_memory：把数据放进锁页内存，拷到 GPU 更快
        "pin_memory": torch.cuda.is_available(),
        # 多进程加载时，每个 epoch 复用 worker，省掉 Windows 上反复启动进程的开销
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader
