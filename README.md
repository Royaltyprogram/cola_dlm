# Cola DLM
**ITS PREVIEW. NOT READY FOR LARGE SCALE TRAINING YET.**

Local, readable reproduction of the Cola DLM architecture described in the
reproduction notes under `docs/reproduction/cola_dlm/`. The code is organized as
a small PyTorch package with typed configs, tiny smoke recipes, training CLIs,
sampling/evaluation helpers, diagnostics, and tests.

This repository is a reproduction scaffold, not an official release. Official
source/model artifacts are tracked separately in
`docs/reproduction/cola_dlm/official_release_compatibility.md` when they become
available.

## Setup

Use the project virtual environment when it exists:

```bash
source "$(git rev-parse --show-toplevel)/myenv/bin/activate"
python -m pip install -e ".[test]"
```

Optional tokenizer support requires:

```bash
python -m pip install -e ".[test,tokenizer]"
```

## Test Status

The codebase is designed around tiny CPU-friendly tests and smoke configs. It
has not been validated by a full paper-scale GPU training run in this workspace.

Useful checks:

```bash
source "$(git rev-parse --show-toplevel)/myenv/bin/activate"
python -m pytest
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Opt-in Modal GPU smoke test:

```bash
source "$(git rev-parse --show-toplevel)/myenv/bin/activate" && modal run scripts/modal_gpu_smoke.py
```

This is a tiny synthetic validation sweep, not a benchmark or training recipe.
It requires Modal authentication and requests a small remote GPU. Normal
`pytest` runs do not require Modal, CUDA, or GPU credentials. A successful run
prints compact JSON fields including CUDA metadata, TextVAE, Stage 2, inference,
loss, and model/tensor device placement checks. The 2026-05-12 Modal attempt is
recorded in `docs/reproduction/cola_dlm/modal_gpu_validation.md`.

If you want to prove the CLI, checkpoint, and sampler path on a GPU, run the
Stage 1 and Stage 2 CLIs for at least one tiny step on a CUDA device, then run
the sampler from the produced checkpoint. The tiny recipes are intended for this
kind of smoke test; the paper-scale recipes are for inspection and parameter
accounting.

## Quick Start

Tiny Stage 1 VAE smoke run:

```bash
train_vae_stage1 \
  --config configs/stage1_tiny_debug.json \
  --data path/to/token_ids.txt \
  --output-dir runs/stage1_tiny
```

Tiny Stage 2 joint VAE-DiT smoke run:

```bash
train_stage2 \
  --config configs/stage2_tiny_debug.json \
  --data path/to/token_ids.txt \
  --vae-checkpoint runs/stage1_tiny/checkpoint.pt \
  --output-dir runs/stage2_tiny
```

Sampling can avoid tokenizer dependencies by passing token ids directly:

```bash
sample \
  --config configs/inference_tiny_debug.json \
  --checkpoint runs/stage2_tiny/checkpoint.pt \
  --prompt-token-ids "1,2,3,4" \
  --output runs/sample.json
```

The console scripts are registered in `pyproject.toml`:

- `train_vae_stage1`
- `train_stage2`
- `sample`
- `evaluate`

## Repository Map

Core package:

- `cola_dlm/config.py`: typed dataclass configs for VAE, DiT, diffusion,
  optimizer, Stage 1, Stage 2, and inference.
- `cola_dlm/config_io.py`: JSON recipe loading/saving, including inherited
  recipes via `extends`.
- `cola_dlm/transformer.py`: shared transformer primitives such as embeddings,
  RMSNorm, feed-forward blocks, RoPE, attention, and a tiny transformer stack.
- `cola_dlm/vae.py`: causal text VAE encoder/decoder, posterior utilities, KL,
  entropy, sampling, and deterministic mode.
- `cola_dlm/block_causal_mask.py`: packed DiT input construction and
  block-causal attention masks.
- `cola_dlm/dit.py`: block-causal text DiT backbone and timestep embedding.
- `cola_dlm/flow_matching.py`: timestep sampling, linear bridge, velocity/x0
  targets, and flow matching loss.
- `cola_dlm/stage1.py`: Stage 1 VAE reconstruction, KL, masking policy, mask
  loss, and one-step pretraining helper.
- `cola_dlm/stage2.py`: frozen reference encoder, Stage 2 VAE/reference/flow
  matching losses, packed DiT wiring, and one-step joint training helper.
- `cola_dlm/inference.py`: prefix encoding, block-wise latent denoising, CFG,
  clean-condition repaint, Euler/Heun sampling, and high-level generation.
- `cola_dlm/tokenizer.py`: narrow tokenizer boundary with Hugging Face adapter
  and deterministic offline fallback.
- `cola_dlm/dataset.py`: tokenized text dataset and chunking utilities.
- `cola_dlm/evaluation.py` and `cola_dlm/evaluate_cli.py`: benchmark prompt
  builders, answer normalization, scoring, and JSONL evaluation CLI.
- `cola_dlm/checkpointing.py`, `cola_dlm/precision.py`, `cola_dlm/logging.py`,
  and `cola_dlm/training_utils.py`: training support utilities.
- `cola_dlm/diagnostics.py`, `cola_dlm/diagnostic_report.py`, and
  `cola_dlm/latent_export.py`: VAE/flow diagnostics, attention-mask reports,
  and optional latent projection export.
- `cola_dlm/parameter_count.py`: meta-device parameter accounting and Markdown
  report generation.

Configs and docs:

- `configs/stage1_tiny_debug.json`: tiny Stage 1 recipe for tests and local
  smoke runs.
- `configs/stage2_tiny_debug.json`: tiny Stage 2 recipe for tests and local
  smoke runs.
- `configs/inference_tiny_debug.json`: tiny inference recipe.
- `configs/stage1_paper.json` and `configs/stage2_paper.json`: stable
  paper-scale entry points for review, not local smoke training.
- `configs/paper/stage2_paper_base.json`: inherited paper-scale Stage 2 base.
- `configs/ablations/`: small paper-ablation variants.
- `configs/README.md`: recipe format and inheritance details.
- `docs/reproduction/cola_dlm/00_context.md`: paper-derived architecture,
  losses, training setup, inference behavior, and open questions.
- `docs/reproduction/cola_dlm/02_pr_plan.md`: implementation plan that produced
  the current PR-unit structure.
- `docs/reproduction/cola_dlm/paper_scale_config.md`: paper-scale defaults and
  memory-planning notes.
- `docs/reproduction/cola_dlm/parameter_counts.md`: generated parameter-count
  report.
- `docs/reproduction/cola_dlm/official_release_compatibility.md`: official
  release reconciliation status.

Tests:

- `tests/conftest.py`: tiny config fixtures used across the suite.
- `tests/test_transformer.py`, `tests/test_vae.py`, `tests/test_dit.py`,
  `tests/test_block_causal_mask.py`: model component behavior.
- `tests/test_flow_matching.py`, `tests/test_stage1.py`,
  `tests/test_stage2.py`: objective and training-step behavior.
- `tests/test_inference.py`, `tests/test_sample_cli.py`: generation behavior and
  sampler CLI.
- `tests/test_train_vae_stage1_cli.py`, `tests/test_train_stage2_cli.py`:
  training CLI smoke coverage.
- `tests/test_config*.py`, `tests/test_ablation_configs.py`,
  `tests/test_parameter_count.py`: config, recipes, and report checks.
- `tests/test_evaluation.py`, `tests/test_evaluate_cli.py`: benchmark prompt and
  scoring behavior.

## Main Data Flow

Stage 1 trains the causal text VAE:

1. `TokenizedTextDataset` yields fixed-length token chunks.
2. `TextVAEEncoder` maps tokens to per-token posterior latents.
3. `TextVAEDecoder` reconstructs tokens from sampled or deterministic latents.
4. `compute_stage1_vae_loss` combines reconstruction, KL, and optional masked
   token auxiliary loss.
5. `stage1_pretraining_step` performs one optimizer step and returns scalar
   diagnostics.

Stage 2 jointly trains the VAE and DiT:

1. A frozen reference encoder is created with `create_frozen_reference_encoder`.
2. The trainable VAE produces clean latents and reconstruction terms.
3. `build_packed_dit_inputs` constructs clean-history/noisy-target packed
   latent sequences.
4. `build_block_causal_attention_mask` enforces block-causal DiT attention.
5. `BlockCausalTextDiT` predicts the configured flow matching target.
6. `compute_stage2_loss` combines VAE, reference KL, and flow matching losses.
7. `stage2_joint_training_step` updates only trainable VAE and DiT parameters.

Inference:

1. `encode_prefix_latents` encodes the known prompt prefix.
2. `iter_generation_blocks` splits generation into block-sized latent spans.
3. `sample_latent_blocks` denoises unknown block latents with Euler or Heun.
4. CFG is applied with `uncond + cfg_scale * (cond - uncond)`.
5. `apply_clean_condition_repaint` restores known prefix latents when requested.
6. `generate` decodes the final latent sequence into token logits and ids.

## Config Patterns

Paper-scale defaults live in the dataclasses and paper recipes, while tests and
local smoke runs override them with tiny dimensions. A typical tiny Stage 2
config looks like this:

```python
from cola_dlm.config import (
    DiTConfig,
    DiffusionConfig,
    OptimizerConfig,
    Stage2Config,
    VAEConfig,
)

tiny_vae = VAEConfig(
    tokenizer_name="tiny",
    vocab_size=128,
    sequence_length=16,
    latent_dim=4,
    encoder_layers=1,
    decoder_layers=1,
    hidden_size=32,
    ffn_size=64,
    num_attention_heads=4,
    attention_head_dim=8,
)
tiny_dit = DiTConfig(
    sequence_length=16,
    latent_dim=4,
    block_size=4,
    num_layers=2,
    hidden_size=32,
    ffn_size=64,
    num_attention_heads=4,
    attention_head_dim=8,
)
tiny_config = Stage2Config(
    vae=tiny_vae,
    dit=tiny_dit,
    diffusion=DiffusionConfig(logit_normal_loc=0.0, logit_normal_scale=0.5),
    optimizer=OptimizerConfig(peak_lr=1.0e-4, warmup_steps=2),
    global_batch_size=2,
    tokens_per_step=32,
)
```

Use `cola_dlm.config_io.load_config` for JSON recipes:

```python
from cola_dlm.config import Stage2Config
from cola_dlm.config_io import load_config

loaded = load_config("configs/stage2_tiny_debug.json", Stage2Config)
config = loaded.config
metadata = loaded.metadata
```

## Practical Development Notes

- Keep implementation changes small and close to the existing module boundary.
- Prefer the tiny configs and focused tests before trying larger runs.
- Treat `docs/reproduction/cola_dlm/00_context.md` as the paper-derived source
  of truth for unresolved behavior until official artifacts are available.
- Do not vendor-copy official source code into this repository. Record official
  comparisons in `official_release_compatibility.md` and add narrow adapters or
  config flags only when they clarify an ambiguity.
- `configs/stage*_paper.json` are useful for shape review and reports; they are
  not proof that paper-scale training fits the local hardware.
