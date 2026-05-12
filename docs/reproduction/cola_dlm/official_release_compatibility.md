# Cola DLM Official Release Compatibility

This note tracks official-release provenance for the reconciliation plan that
starts from [00_context.md](00_context.md). It is intentionally short until the
official source and model artifacts are publicly available.

## Provenance Check

Checked on: 2026-05-12 05:31 KST (+0900)

| Artifact | Official link | Status | Revision identifiers |
| --- | --- | --- | --- |
| Source code | <https://github.com/ByteDance-Seed/Cola-DLM> | Unavailable: the project link did not resolve to a public repository during this check. | Git commit hash: unavailable; release tag: unavailable. |
| Model artifacts | <https://huggingface.co/ByteDance-Seed/Cola-DLM> | Unavailable: the model link did not resolve to a public Hugging Face model during this check. | Model revision: unavailable; model/config filenames: unavailable. |

No official source files or model files were copied into this repository. The
next reconciliation step is blocked until the source commit and model revision
can be recorded.

## Reconciliation Checklist

| Area | Local targets | Official reference | Status |
| --- | --- | --- | --- |
| Model shapes and config values | `cola_dlm/config.py`, `configs/` | Unavailable | Blocked |
| VAE masks and masking objective | `cola_dlm/vae.py`, `cola_dlm/stage1.py` | Unavailable | Blocked |
| DiT block-causal packing and attention mask | `cola_dlm/block_causal_mask.py`, `cola_dlm/dit.py` | Unavailable | Blocked |
| Stage 1 and Stage 2 loss parameterization | `cola_dlm/stage1.py`, `cola_dlm/stage2.py` | Unavailable | Blocked |
| Flow Matching prediction type and noise schedule | `cola_dlm/flow_matching.py` | Unavailable | Blocked |
| CFG training dropout, unconditional construction, and inference formula | `cola_dlm/stage2.py`, `cola_dlm/inference.py` | Unavailable | Blocked |
| First-block generation / repaint behavior | `cola_dlm/inference.py` | Unavailable | Blocked |

## Verification

- Official revision identifiers are recorded as unavailable rather than guessed.
- This note links back to [00_context.md](00_context.md).
- No code or test files changed in this unit, so the test suite was not run.
