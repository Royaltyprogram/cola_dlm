# Modal GPU Validation Result

## Run Metadata

- Initial blocked attempt timestamp: 2026-05-12 14:12:52 KST +0900
- Successful retry timestamp: 2026-05-12 14:18:29 KST +0900
- Branch: `codex/modal-gpu-validation-run`
- Modal run URL:
  `https://modal.com/apps/yechansub/main/ap-f3kKXNLinFPASYrY4MN5eR`
- Command:

  ```bash
  source "$(git rev-parse --show-toplevel)/myenv/bin/activate" && modal run scripts/modal_gpu_smoke.py
  ```

## Modal Result

The first attempt failed before remote execution with a transient Modal server
connectivity error:

```text
Could not connect to the Modal server.
```

The command was retried from the repository root and completed successfully on
an actual Modal GPU container. The validation is still a tiny synthetic sweep,
not a benchmark or real training run.

## Remote GPU Fields

- Actual Modal GPU run: yes
- Overall success: true
- CUDA available: true
- CUDA device count: 1
- Current CUDA device index: 0
- GPU name: `Tesla T4`
- Selected validation device: `cuda`
- Limitations: none

## Remote Validation Checks

CUDA metadata:

- Success: true
- Device name: `Tesla T4`
- Device count: 1
- Current device index: 0

TextVAE training check:

- Success: true
- Loss finite: true
- Loss: `3.896153450012207`
- At least one trainable parameter changed: true
- Devices match CUDA: true
- Loss tensor device: `cuda:0`
- VAE parameter device: `cuda:0`
- Token device: `cuda:0`
- Mask device: `cuda:0`
- Posterior device: `cuda:0`
- Posterior mean device: `cuda:0`
- Posterior logvar device: `cuda:0`
- Latent device: `cuda:0`
- Logits device: `cuda:0`

Stage 2 joint training check:

- Success: true
- Steps: 1
- Total loss finite: true
- Total loss: `10.098993301391602`
- At least one trainable parameter changed: true
- Devices match CUDA: true
- Loss tensor device: `cuda:0`
- VAE parameter device: `cuda:0`
- Reference encoder parameter device: `cuda:0`
- DiT parameter device: `cuda:0`
- Token device: `cuda:0`
- Mask device: `cuda:0`

Inference/generate check:

- Success: true
- Skipped: false
- Devices match CUDA: true
- Generated latent device: `cuda:0`
- All-latent device: `cuda:0`
- Response logits device: `cuda:0`
- Response token device: `cuda:0`
- Token device: `cuda:0`
- Mask device: `cuda:0`
- VAE parameter device: `cuda:0`
- DiT parameter device: `cuda:0`

## Raw Modal JSON

```json
{"checks":{"inference_generate":{"all_latent_device":"cuda:0","devices_match":true,"dit_parameter_device":"cuda:0","generated_latent_device":"cuda:0","mask_device":"cuda:0","response_logits_device":"cuda:0","response_token_device":"cuda:0","skipped":false,"success":true,"token_device":"cuda:0","vae_parameter_device":"cuda:0"},"stage2_joint_training":{"devices_match":true,"dit_parameter_device":"cuda:0","loss_finite":true,"loss_tensor_device":"cuda:0","mask_device":"cuda:0","reference_encoder_parameter_device":"cuda:0","steps":1,"success":true,"token_device":"cuda:0","total_loss":10.098993301391602,"trainable_parameter_changed":true,"vae_parameter_device":"cuda:0"},"text_vae":{"devices_match":true,"latent_device":"cuda:0","logits_device":"cuda:0","loss":3.896153450012207,"loss_finite":true,"loss_tensor_device":"cuda:0","mask_device":"cuda:0","posterior_device":"cuda:0","posterior_logvar_device":"cuda:0","posterior_mu_device":"cuda:0","success":true,"token_device":"cuda:0","trainable_parameter_changed":true,"vae_parameter_device":"cuda:0"}},"cuda_available":true,"cuda_metadata":{"available":true,"current_device_index":0,"device_count":1,"device_name":"Tesla T4"},"device":"cuda","limitations":[],"success":true}
```

## Local CPU-Only Verification

These checks ran separately from Modal and do not replace the remote GPU result.

Focused smoke tests:

```bash
source "$(git rev-parse --show-toplevel)/myenv/bin/activate" && python -m pytest tests/test_modal_gpu_smoke.py
```

Result from the Crack execution: `9 passed, 1 warning in 1.76s`.

Local CPU validation helper sample:

- Overall success: true
- Device: `cpu`
- CUDA available locally: false
- Limitations: none
- TextVAE success: true
- TextVAE loss finite: true
- TextVAE loss: `4.296713352203369`
- Stage 2 joint training success: true
- Stage 2 total loss finite: true
- Stage 2 total loss: `8.565595626831055`
- Inference/generate success: true
- Inference/generate skipped: false

Full local test suite:

```bash
source "$(git rev-parse --show-toplevel)/myenv/bin/activate" && python -m pytest
```

Result from the Crack execution: `3 failed, 310 passed, 1 warning in 2.09s`.

The failures were all in `tests/test_train_vae_stage1_cli.py` and raised:

```text
ValueError: 2D attention_mask must be shaped [seq, seq]
```

## Limitations

- This validates tiny synthetic GPU execution paths only.
- It does not claim paper-scale training, benchmark performance, checkpoint
  compatibility, or dataset-level correctness.
- The full local suite still has Stage 1 CLI failures outside this Modal GPU
  validation result.
