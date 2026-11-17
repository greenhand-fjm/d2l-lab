"""实验配置：用一个 dataclass 描述一次实验的全部超参数。

工程要点
- 配置和代码分离：改超参数只改 YAML 文件或命令行，不改代码。
- 尽早报错：拼错的键名、类型不对的值，在加载配置时就报错，
  而不是训练跑了半小时才崩。
- 可追溯：每次运行都会把最终配置存进 run 目录，结果永远能对应到参数。
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(name: str, value: Any, typ: Any) -> Any:
    """把 YAML / 命令行读进来的值转换成字段声明的类型。"""
    try:
        if typ is bool:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ValueError
        if typ is int:
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            return int(value)
        if typ in (float, str):
            return typ(value)
        # 其余字段（model_args）是字典；命令行传进来时是 YAML 文本
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (TypeError, ValueError):
        type_name = getattr(typ, "__name__", str(typ))
        raise ValueError(f"config '{name}': cannot use {value!r} as {type_name}") from None


@dataclass
class Config:
    name: str = "experiment"
    # 数据
    dataset: str = "synthetic_linear"
    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 0
    # 任务与模型
    task: str = "regression"  # regression | classification
    model: str = "linear"
    model_args: dict[str, Any] = field(default_factory=dict)
    # 优化
    epochs: int = 5
    lr: float = 0.03
    optimizer: str = "sgd"  # sgd | adam | adamw
    weight_decay: float = 0.0
    # 运行
    seed: int = 42
    device: str = "auto"  # auto | cuda | cpu
    amp: bool = False  # bf16 自动混合精度，只在 GPU 上生效
    output_dir: str = "runs"
    resume: bool = True  # 发现 last.pt 时从断点继续
    log_every: int = 50

    def __post_init__(self) -> None:
        for f in fields(self):
            setattr(self, f.name, _coerce(f.name, getattr(self, f.name), f.type))
        if self.task not in {"regression", "classification"}:
            raise ValueError(f"task must be 'regression' or 'classification', got {self.task!r}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data).with_overrides(overrides or [])

    def with_overrides(self, items: list[str]) -> "Config":
        """应用命令行里的 KEY=VALUE，例如 --set lr=0.1 --set epochs=10。"""
        data = asdict(self)
        for item in items:
            key, sep, raw = item.partition("=")
            key = key.strip()
            if not sep or key not in data:
                raise ValueError(
                    f"bad override {item!r}: expected KEY=VALUE, KEY in {sorted(data)}"
                )
            data[key] = raw.strip()
        return Config.from_dict(data)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True, sort_keys=False)
