import pytest

from d2l_lab.config import Config


def test_yaml_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: t\nlr: 1e-3\nepochs: 2\n", encoding="utf-8")
    cfg = Config.from_yaml(path, ["epochs=4", "amp=true", "model_args={in_features: 3}"])
    # 坑：YAML 会把 1e-3 读成字符串（它要求写成 1.0e-3），配置类负责转成 float
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.epochs == 4
    assert cfg.amp is True
    assert cfg.model_args == {"in_features": 3}


def test_unknown_key_is_rejected():
    # 把 lr 拼成 learning_rate 时应该立刻报错，而不是悄悄用默认值训练
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"learning_rate": 0.1})


def test_bad_value_is_rejected():
    with pytest.raises(ValueError):
        Config(epochs="three")
    with pytest.raises(ValueError):
        Config(task="segmentation")


def test_save_round_trip(tmp_path):
    cfg = Config(name="rt", lr=0.5, model_args={"hidden": [64, 32]})
    cfg.save(tmp_path / "config.yaml")
    assert Config.from_yaml(tmp_path / "config.yaml") == cfg
