"""Pre-LayerNorm Transformer encoder block.

Wang et al. (2019) showed Pre-Norm placement (LayerNorm before each sub-layer,
rather than after) yields more stable gradients and avoids the early-training
loss spikes that a Post-Norm setup exhibited in preliminary runs -- this is the
justification recorded in the accompanying research report (Sec. 4.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """A single Pre-Norm Transformer encoder block.

    Sub-layers: Multi-Head Self-Attention and a position-wise Feed-Forward
    Network. Each sub-layer is preceded by LayerNorm and wrapped by a residual
    connection:

    .. math::
        z' &= z^{l-1} + \\mathrm{MHA}(\\mathrm{LN}_1(z^{l-1})) \\\\
        z^{l} &= z' + \\mathrm{FFN}(\\mathrm{LN}_2(z'))

    Parameters
    ----------
    d_token:
        Token embedding dimension (must be divisible by ``n_heads``).
    n_heads:
        Number of self-attention heads.
    ffn_factor:
        FFN hidden-dimension expansion factor. The FT-Transformer paper uses
        ``floor(d_token * 4/3)`` rather than the standard 4x Transformer
        expansion, appropriate for the short (F+1)-length sequences here.
    dropout:
        Dropout probability applied inside attention and the FFN.
    """

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        ffn_factor: float = 1.333,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) must be divisible by n_heads ({n_heads})")
        self.norm1 = nn.LayerNorm(d_token)  # pre-norm for attention
        # Multi-head self-attention (batch_first: B, T, d convention)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)  # pre-norm for FFN
        d_ff = int(d_token * ffn_factor)  # hidden dim of FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Pre-Norm attention + FFN block.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, T, d_token)`` -- sequence of token embeddings.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, T, d_token)``.
        """
        # Pre-Norm Attention + residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)  # self-attention
        x = x + attn_out

        # Pre-Norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x
