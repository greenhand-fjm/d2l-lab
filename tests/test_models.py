import pytest
import torch

from d2l_lab.models import build_model


def test_linear_output_shape():
    model = build_model("linear", in_features=2, out_features=1)
    assert model(torch.randn(8, 2)).shape == (8, 1)


def test_mlp_accepts_images():
    model = build_model("mlp", hidden=[64, 32])
    assert model(torch.randn(8, 1, 28, 28)).shape == (8, 10)


def test_unknown_model_lists_choices():
    with pytest.raises(KeyError, match="available"):
        build_model("transformer9000")
