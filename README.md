# d2l-lab

把《动手学深度学习》里的模型写成一个工程化的项目：配置驱动、可复现、能断点续训、有测试。

## 目录结构

```
d2l-lab/
├── configs/              # 每个实验一个 YAML，改超参数只改这里
│   ├── linreg.yaml       # 第 3 章 线性回归
│   └── mlp_fashion.yaml  # 第 5 章 多层感知机
├── src/d2l_lab/
│   ├── config.py         # 配置类：加载、校验、命令行覆盖
│   ├── data.py           # 数据集注册表
│   ├── models.py         # 模型注册表
│   ├── trainer.py        # 通用训练循环 + 断点续训
│   ├── utils.py          # 随机种子、设备、日志
│   └── cli.py            # 命令行入口
├── tests/                # pytest 测试
├── data/                 # 数据集（自动下载，不进 git）
└── runs/<实验名>/         # 每次运行的产物（不进 git）
    ├── config.yaml       #   这次实际用的配置
    ├── train.log         #   完整日志
    ├── metrics.jsonl     #   每个 epoch 的指标
    ├── last.pt           #   最新断点
    └── best.pt           #   验证集最好的模型
```

## 安装

在 Anaconda Prompt 里：

```
conda activate pytorch
cd C:\Users\LEGION\Projects\d2l-lab
pip install -e ".[dev]"
```

`-e` 是可编辑安装：改了 `src/` 里的代码立即生效，不用重装。

## 训练

```
python -m d2l_lab.cli --config configs/linreg.yaml
python -m d2l_lab.cli --config configs/linreg.yaml --set lr=0.1 --set epochs=10
python -m d2l_lab.cli --config configs/mlp_fashion.yaml
```

- 中途中断（Ctrl+C、合盖、断电）后，重新运行同一条命令会从 `last.pt` 接着训练。
- 想从头训练：加 `--no-resume`。
- 想做对比实验：用 `--set name=新名字` 换一个输出目录，旧结果不会被覆盖。

## 测试与代码检查

```
pytest            # 运行所有测试，几秒钟
ruff check .      # 检查代码风格和常见错误
ruff format .     # 自动格式化
```

## 加一个新模型（学完一章后）

1. 在 `models.py` 写一个 `nn.Module`，加上 `@register_model("名字")`。
2. 在 `tests/test_models.py` 加一个测试：输入一批假数据，检查输出形状。
3. 复制一个 `configs/*.yaml`，把 `model` 改成新名字。
4. `pytest` 通过后再开始真正训练。

训练循环、日志、断点续训都不用改。
