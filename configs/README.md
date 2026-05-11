# Config Recipes

Recipe files are JSON so they can be loaded with the Python standard library.
Each recipe stores the typed model config under `config`. Other top-level keys
are preserved as run metadata for later CLI entrypoints.

```python
from cola_dlm.config import Stage1Config
from cola_dlm.config_io import load_config

loaded = load_config("configs/stage1_tiny_debug.json", Stage1Config)
stage1_config = loaded.config
run_metadata = loaded.metadata
```

The tiny-debug recipes use the same small dimensions as the pytest fixtures and
set only a couple of local training steps.

Recipes may optionally declare a top-level `extends` path. The path is resolved
relative to the recipe that declares it. Inherited `config` objects are merged
deeply, while scalar and list values replace the base value. Top-level metadata
is also inherited, with the child recipe winning on key conflicts.

The stable paper-scale entry points are:

- `configs/stage1_paper.json`
- `configs/stage2_paper.json`

`configs/stage2_paper.json` intentionally contains no behavioral overrides; it
extends `configs/paper/stage2_paper_base.json`, which holds the paper-scale VAE,
DiT, diffusion, optimizer, batch, and token-count defaults traced to
`docs/reproduction/cola_dlm/00_context.md`.

The paper-scale recipes are for configuration review, reporting, and generated
parameter-count audits in this repo. They are not supported local training or
smoke-test recipes. See
`docs/reproduction/cola_dlm/paper_scale_config.md` for the paper defaults and
memory-planning notes.

Sampling can use `--prompt-token-ids` directly, which is the preferred smoke-test
path because it avoids tokenizer dependencies. `--prompt` is available for local
experiments, but its offline fallback is a deterministic byte tokenizer and does
not reproduce OLMo-compatible token ids.
