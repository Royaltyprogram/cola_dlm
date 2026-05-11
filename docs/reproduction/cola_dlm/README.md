# Cola DLM Reproduction Notes

This directory contains the first-pass reproduction context for:

- arXiv paper: <https://arxiv.org/html/2605.06548v1>
- project page: <https://hongcanguo.github.io/Cola-DLM/>

Files:

- `00_context.md`: architecture, losses, training setup, inference, unresolved implementation details.
- `01_figures.md`: downloaded figure inventory and notes on which figures matter for implementation.
- `02_pr_plan.md`: large implementation PR sequence for reproducing Cola DLM.
- `figures/`: copied/downloaded figure assets.
- `source_cache/`: raw HTML and arXiv source cache used for extraction.

Recommended next step:

1. Create a clean PyTorch architecture skeleton from `00_context.md`.
2. Implement the block-causal packed attention mask first, because it determines the DiT training interface.
3. Add Stage 1/Stage 2 loss functions with unresolved weights kept configurable.
