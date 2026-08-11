"""Unit tests for :class:`ft_transformer.blocks.TransformerBlock`."""

from __future__ import annotations

import pytest
import torch

from ft_transformer.blocks import TransformerBlock


def test_output_shape_preserved():
    blk = TransformerBlock(d_token=64, n_heads=4)
    x = torch.randn(32, 9, 64)  # batch=32, seq_len=9 (8 features + CLS)
    out = blk(x)
    assert out.shape == (32, 9, 64)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_various_head_counts(n_heads):
    d_token = 32
    blk = TransformerBlock(d_token=d_token, n_heads=n_heads)
    x = torch.randn(8, 5, d_token)
    out = blk(x)
    assert out.shape == x.shape


def test_invalid_head_count_raises():
    with pytest.raises(ValueError):
        TransformerBlock(d_token=63, n_heads=4)  # 63 not divisible by 4


def test_gradients_flow():
    blk = TransformerBlock(d_token=32, n_heads=4)
    x = torch.randn(4, 6, 32, requires_grad=True)
    out = blk(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_zero_dropout_is_reproducible_in_eval_mode():
    """In eval mode (dropout disabled), the same input must give the same output."""
    blk = TransformerBlock(d_token=32, n_heads=4, dropout=0.5)
    blk.eval()
    x = torch.randn(4, 6, 32)
    with torch.no_grad():
        out1 = blk(x)
        out2 = blk(x)
    assert torch.allclose(out1, out2)
