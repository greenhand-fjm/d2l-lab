"""模型注册表：配置文件里写模型名字和参数，build_model 负责创建。

学完新的一章，就在这里加一个类、加一行 @register_model。
"""

from collections.abc import Callable, Sequence

import torch
from torch import nn

_MODELS: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    def decorator(cls: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _MODELS:
            raise ValueError(f"model {name!r} registered twice")
        _MODELS[name] = cls
        return cls

    return decorator


@register_model("linear")
class LinearRegression(nn.Module):
    """D2L 3.5：线性回归的简洁实现。"""

    def __init__(self, in_features: int = 2, out_features: int = 1) -> None:
        super().__init__()
        self.net = nn.Linear(in_features, out_features)
        nn.init.normal_(self.net.weight, std=0.01)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("mlp")
class MLP(nn.Module):
    """D2L 5.2：多层感知机。hidden 是每个隐藏层的宽度，例如 [256] 或 [256, 128]。"""

    def __init__(
        self,
        in_features: int = 784,
        hidden: Sequence[int] = (256,),
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_MODELS)}")
    return _MODELS[name](**kwargs)
