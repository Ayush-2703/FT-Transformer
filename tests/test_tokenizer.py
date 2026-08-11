"""Unit tests for :class:`ft_transformer.tokenizer.NumericalTokenizer`."""

from __future__ import annotations

import torch

from ft_transformer.tokenizer import NumericalTokenizer


def test_output_shape():
    tok = NumericalTokenizer(n_features=8, d_token=64)
    x = torch.randn(32, 8)
    out = tok(x)
    assert out.shape == (32, 8, 64)


def test_output_is_finite():
    tok = NumericalTokenizer(n_features=8, d_token=64)
    x = torch.randn(16, 8)
    out = tok(x)
    assert torch.isfinite(out).all()


def test_zero_input_gives_bias():
    """With x = 0, token_i = W_i * 0 + b_i = b_i exactly."""
    tok = NumericalTokenizer(n_features=5, d_token=8)
    x = torch.zeros(3, 5)
    out = tok(x)
    expected = tok.b.unsqueeze(0).expand(3, -1, -1)
    assert torch.allclose(out, expected)


def test_each_feature_has_independent_parameters():
    """Different features should get different learned weight/bias vectors."""
    tok = NumericalTokenizer(n_features=8, d_token=16)
    assert not torch.allclose(tok.W[0], tok.W[1])


def test_gradients_flow():
    tok = NumericalTokenizer(n_features=8, d_token=16)
    x = torch.randn(4, 8, requires_grad=True)
    out = tok(x)
    out.sum().backward()
    assert tok.W.grad is not None
    assert tok.b.grad is not None
    assert torch.isfinite(tok.W.grad).all()
