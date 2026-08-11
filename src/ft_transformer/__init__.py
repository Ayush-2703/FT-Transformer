"""FT-Transformer: Feature Tokenizer + Transformer for tabular regression.

A from-scratch PyTorch implementation of Gorishniy et al. (2021), evaluated
against classical baselines on the California Housing dataset.
"""

from ft_transformer.blocks import TransformerBlock
from ft_transformer.model import FTTransformer
from ft_transformer.tokenizer import NumericalTokenizer

__version__ = "1.0.0"
__all__ = ["NumericalTokenizer", "TransformerBlock", "FTTransformer"]
