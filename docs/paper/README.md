# Research report

`FT-Transformer-Report.pdf` / `.docx` is the internship research report this repository is built
from: dataset analysis, full architectural derivation (with equations), design-decision
rationale (Table 4.2), and the complete results section.

Two things in the packaged code here intentionally diverge from the report text, both covered by
tests and explained in the top-level [`README.md`](../../README.md#results) and
[`CHANGELOG.md`](../../CHANGELOG.md):

1. **Default-config parameter count.** The report's Table 5.2 lists 184,321; the notebook's own
   printed output (and this repo's model) give 57,131. The code/notebook figure is treated as
   ground truth.
2. **Permutation-invariance claim.** The report states shuffling input feature order gives
   identical predictions. What's actually true is invariance to *token sequence order after
   tokenization* — not to raw input column order, since the tokenizer learns independent
   parameters per feature index. See `tests/test_model.py`.

Everything else — the architecture, training configuration, ablation results, and Optuna search
results — is reproduced as-is from the report and verified against the source notebook.
