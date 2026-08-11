"""Full Feature Tokenizer + Transformer model for tabular regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from ft_transformer.blocks import TransformerBlock
from ft_transformer.tokenizer import NumericalTokenizer


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular regression.

    Architecture
    ------------
    ``NumericalTokenizer -> CLS prepend -> N x TransformerBlock ->
    LayerNorm(CLS) -> Linear(1)``

    No positional encoding is used: feature order in the tokenised sequence
    carries no sequential meaning, so the model is permutation-equivariant with
    respect to feature order (up to the fixed CLS position) -- see
    :func:`test_model.test_permutation_invariance` for a direct check of this
    property.

    Parameters
    ----------
    n_features:
        Number of input features (8 for California Housing).
    d_token:
        Per-feature token embedding dimension.
    n_blocks:
        Number of stacked :class:`~ft_transformer.blocks.TransformerBlock` layers.
    n_heads:
        Number of self-attention heads per block.
    dropout:
        Dropout probability used throughout attention and FFN sub-layers.
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_blocks: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericalTokenizer(n_features, d_token)
        # Learnable CLS token -- shared parameter, expanded per-batch in forward()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_token, n_heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_token)  # final norm on CLS representation
        self.head = nn.Linear(d_token, 1)  # regression head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a scalar regression target for each row.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, n_features)`` -- batch of standardised feature
            vectors.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B,)`` -- scalar regression output.
        """
        tokens = self.tokenizer(x)  # (B, F, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, d)
        for block in self.blocks:
            tokens = block(tokens)  # (B, F+1, d)
        cls_repr = self.norm(tokens[:, 0])  # (B, d)
        return self.head(cls_repr).squeeze(-1)  # (B,)
