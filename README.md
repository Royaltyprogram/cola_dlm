# Cola DLM

Local reproduction scaffold for Cola DLM. This repository currently defines the
package skeleton, typed configuration objects, and CPU-only smoke tests that
later model and training PRs can build on.

This PR intentionally contains no model implementation and no real training
loops.

## Setup

```bash
python -m pip install -e ".[test]"
```

## Test

```bash
python -m pytest
```

## Tiny Config Example

Paper-scale defaults are encoded in `cola_dlm/config.py`, based on
`docs/reproduction/cola_dlm/00_context.md`. For smoke tests, override those
defaults by passing small nested configs explicitly:

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

The pytest fixtures in `tests/conftest.py` show the same override pattern for
Stage 1, Stage 2, inference, and individual sub-configs.
