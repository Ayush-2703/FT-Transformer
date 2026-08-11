"""Numerical Feature Tokenizer.

Projects each scalar tabular feature into a ``d_token``-dimensional embedding via
an independent, learnable, per-feature affine map. This is the representational
bridge that lets self-attention -- designed for sequences of semantically rich
word embeddings -- operate meaningfully over raw scalar tabular features, which
otherwise carry no inherent geometric proximity (Gorishniy et al., 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NumericalTokenizer(nn.Module):
    """Tokenise every numerical feature into a ``d_token``-dim vector.

    For a standardised scalar feature value :math:`x_i`, the token is

    .. math::
        t_i = W_i \\cdot x_i + b_i, \\qquad W_i, b_i \\in \\mathbb{R}^{d}

    with an independent weight/bias pair learned per feature index ``i``.

    Parameters
    ----------
    n_features:
        Number of input features (8 for California Housing).
    d_token:
        Embedding dimension per feature.
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        # Per-feature weight: shape (F, d_token)
        self.W = nn.Parameter(torch.randn(n_features, d_token))
        # Per-feature bias:   shape (F, d_token)
        self.b = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenise a batch of flat feature vectors.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, F)`` -- flat, standardised feature vectors.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, F, d_token)`` -- one token embedding per feature.
        """
        # x.unsqueeze(-1) -> (B, F, 1); broadcast-multiply with W (F, d)
        return x.unsqueeze(-1) * self.W + self.b
