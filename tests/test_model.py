"""Unit tests for :class:`ft_transformer.model.FTTransformer`."""

from __future__ import annotations

import pytest
import torch

from ft_transformer.model import FTTransformer
from ft_transformer.utils import count_trainable_parameters


def test_output_shape():
    model = FTTransformer(n_features=8, d_token=64, n_blocks=2, n_heads=4)
    x = torch.randn(32, 8)
    out = model(x)
    assert out.shape == (32,)


@pytest.mark.parametrize("n_blocks,n_heads", [(1, 1), (1, 4), (2, 4), (3, 8)])
def test_various_configs_run(n_blocks, n_heads):
    model = FTTransformer(n_features=8, d_token=32, n_blocks=n_blocks, n_heads=n_heads)
    x = torch.randn(10, 8)
    out = model(x)
    assert out.shape == (10,)
    assert torch.isfinite(out).all()


def test_default_config_parameter_count_matches_reference():
    """Default config (d=64, blocks=2, heads=4) should have 57,131 trainable params,
    exactly matching the parameter count printed in Cell 6 of the research notebook."""
    model = FTTransformer(n_features=8, d_token=64, n_blocks=2, n_heads=4)
    n_params = count_trainable_parameters(model)
    assert n_params == 57_131


def test_cls_output_invariant_to_token_sequence_order():
    """No positional encoding is used, so the CLS output must be invariant to the
    *order* in which already-computed feature tokens are concatenated into the
    sequence -- attention aggregates over the token multiset with no notion of
    position. Verified by tokenizing once, then feeding the transformer blocks the
    same tokens in two different sequence orders and checking the CLS read-out
    matches.

    Note this is a distinct (and more precise) claim than "shuffling the raw input
    *columns* gives the same prediction": it does not, because
    :class:`~ft_transformer.tokenizer.NumericalTokenizer` intentionally learns an
    independent weight/bias pair *per feature index* (Sec. 4.3 of the report,
    "maintaining independent embedding parameters for each predictor like MedInc
    and Latitude"). See `test_raw_feature_column_permutation_changes_prediction`
    below for a direct check of that companion behaviour.
    """
    model = FTTransformer(n_features=8, d_token=32, n_blocks=2, n_heads=4, dropout=0.0)
    model.eval()

    x = torch.randn(5, 8)
    perm = torch.randperm(8)

    with torch.no_grad():
        tokens = model.tokenizer(x)  # (B, F, d) -- correct per-feature tokens
        cls = model.cls_token.expand(x.size(0), -1, -1)

        seq = torch.cat([cls, tokens], dim=1)
        seq_reordered = torch.cat([cls, tokens[:, perm, :]], dim=1)

        for block in model.blocks:
            seq = block(seq)
            seq_reordered = block(seq_reordered)

        out = model.head(model.norm(seq[:, 0])).squeeze(-1)
        out_reordered = model.head(model.norm(seq_reordered[:, 0])).squeeze(-1)

    assert torch.allclose(out, out_reordered, atol=1e-5, rtol=1e-4)


def test_raw_feature_column_permutation_changes_prediction():
    """Complementary to the test above: shuffling the *raw input columns* (before
    tokenization) generally changes the prediction, because each feature slot owns
    distinct learned tokenizer parameters. This is intentional -- it is what lets
    the model learn that, e.g., feature index 0 (MedInc) behaves differently from
    feature index 6 (Latitude) -- but it means the model is not a symmetric
    function of the raw feature vector, only of the (already-identified) token
    set. This test documents that behaviour explicitly."""
    model = FTTransformer(n_features=8, d_token=32, n_blocks=2, n_heads=4, dropout=0.0)
    model.eval()

    torch.manual_seed(123)
    x = torch.randn(5, 8)
    perm = torch.randperm(8)
    while torch.equal(perm, torch.arange(8)):  # ensure a non-trivial permutation
        perm = torch.randperm(8)
    x_permuted = x[:, perm]

    with torch.no_grad():
        out_original = model(x)
        out_permuted = model(x_permuted)

    assert not torch.allclose(out_original, out_permuted, atol=1e-4)


def test_gradients_flow_through_full_model():
    model = FTTransformer(n_features=8, d_token=32, n_blocks=2, n_heads=4)
    x = torch.randn(6, 8)
    target = torch.randn(6)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


def test_invalid_head_dimension_raises():
    with pytest.raises(ValueError):
        FTTransformer(n_features=8, d_token=50, n_blocks=1, n_heads=3)
