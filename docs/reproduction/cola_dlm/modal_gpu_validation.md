# Modal GPU Validation Result

## Run Metadata

- Timestamp: 2026-05-12 14:12:52 KST +0900
- Branch: `codex/modal-gpu-validation-run`
- Command:

  ```bash
  source "$(git rev-parse --show-toplevel)/myenv/bin/activate" && modal run scripts/modal_gpu_smoke.py
  ```

## Modal Result

This was a blocked Modal GPU validation attempt. The command exited with status
1 before remote execution, so no actual Modal GPU run completed and GPU
validation did not pass.

Exact error:

```text
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ Could not connect to the Modal server.                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Remote GPU Fields

These fields were not available because the Modal client could not connect to
the Modal server before starting a remote container.

- Actual Modal GPU run: no
- CUDA available: not observed
- CUDA device count: not observed
- Current CUDA device index: not observed
- GPU name: not observed
- Selected validation device: not observed

## Remote Validation Checks

No remote validation checks ran.

- CUDA metadata success: false
- TextVAE success: false, not run
- Stage 2 joint training success: false, not run
- Inference/generate success: false, not run
- Modal GPU validation success: false

## Remote Loss and Placement Fields

No Modal GPU loss or device-placement values were produced.

- TextVAE loss: not observed
- TextVAE devices: not observed
- Stage 2 total loss: not observed
- Stage 2 devices: not observed
- Inference/generate devices: not observed

## Local CPU-Only Verification

These checks ran separately from Modal and do not prove CUDA or remote GPU
execution.

Focused smoke tests:

```bash
source "$(git rev-parse --show-toplevel)/myenv/bin/activate" && python -m pytest tests/test_modal_gpu_smoke.py
```

Result: `9 passed, 1 warning in 1.76s`.

Local CPU validation helper sample:

- Overall success: true
- Device: `cpu`
- CUDA available locally: false
- Limitations: none
- TextVAE success: true
- TextVAE loss finite: true
- TextVAE loss: `4.296713352203369`
- TextVAE devices: loss tensor `cpu`, VAE parameters `cpu`, tokens `cpu`,
  mask `cpu`, posterior `cpu`, latent `cpu`, logits `cpu`
- Stage 2 joint training success: true
- Stage 2 total loss finite: true
- Stage 2 total loss: `8.565595626831055`
- Stage 2 devices: loss tensor `cpu`, VAE parameters `cpu`, reference encoder
  parameters `cpu`, DiT parameters `cpu`, tokens `cpu`, mask `cpu`
- Inference/generate success: true
- Inference/generate skipped: false
- Inference/generate devices: generated latents `cpu`, all latents `cpu`,
  response logits `cpu`, response tokens `cpu`, tokens `cpu`, mask `cpu`, VAE
  parameters `cpu`, DiT parameters `cpu`

Full local test suite:

```bash
source "$(git rev-parse --show-toplevel)/myenv/bin/activate" && python -m pytest
```

Result: `3 failed, 310 passed, 1 warning in 2.09s`.

The failures were all in `tests/test_train_vae_stage1_cli.py` and raised:

```text
ValueError: 2D attention_mask must be shaped [seq, seq]
```

## Limitations

- The Modal result is blocked by external connectivity to the Modal server.
- CUDA availability, GPU name, remote finite losses, and remote device placement
  were not validated.
- The local CPU-only helper and focused tests passed, but they are not evidence
  of Modal GPU execution.
- The full local suite currently has Stage 1 CLI failures outside this Modal
  GPU result record.
