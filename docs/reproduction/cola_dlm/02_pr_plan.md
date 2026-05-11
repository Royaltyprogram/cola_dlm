# Cola DLM Reproduction PR Plan

This plan is ordered for an architecture-first reproduction. Early PRs should make the model shapes, masks, and losses correct before attempting expensive training.

## PR 1: Project Skeleton and Typed Configs

Goal: create a clean minimal PyTorch package that can hold the reproduction without mixing architecture, training, and experiments.

Scope:

- Add package layout: `cola_dlm/`.
- Add config dataclasses for VAE, DiT, diffusion, optimizer, Stage 1, Stage 2, and inference.
- Add small smoke-test fixtures for tiny model sizes.
- Add basic dependency file and test command.
- Add `README.md` usage notes for the local implementation.

Out of scope:

- Full model implementation.
- Real training loops.

Acceptance criteria:

- `python -m pytest` runs at least config and import tests.
- A tiny config can be instantiated without importing training code.
- No hard-coded paper-scale constants outside config defaults.

## PR 2: Transformer Primitives

Goal: implement reusable transformer components shared by the Text VAE and DiT.

Scope:

- Token embedding and output projection helpers.
- RMSNorm or LayerNorm, matching the chosen simple baseline.
- RoPE implementation.
- Multi-head attention with causal and arbitrary boolean/additive masks.
- Feed-forward block with configurable activation.
- Transformer block with clean residual structure.
- Unit tests for shape, mask behavior, and RoPE dimensions.

Out of scope:

- VAE posterior logic.
- Block-causal DiT packing.

Acceptance criteria:

- Tiny transformer forward pass works on CPU.
- Causal mask test proves future tokens cannot affect previous outputs.
- Arbitrary attention mask test is deterministic and easy to read.

## PR 3: Causal Text VAE Architecture

Goal: express the paper's Text VAE faithfully: strictly causal encoder/decoder, per-token latent posterior, no sequence compression by default.

Scope:

- `TextVAEEncoder`: tokens -> posterior `mu`, `logvar`.
- `TextVAEDecoder`: tokens plus latents -> token logits.
- `DiagonalGaussianPosterior` helper with sample, mode, KL, entropy/log-prob.
- Reparameterization path.
- VAE logSNR utility.
- Patch-size config stub, default `patch_size=1`.
- Unit tests for `[batch, seq, latent_dim]` shape and causal behavior.

Out of scope:

- BERT-style mask loss implementation details beyond method hooks.
- Stage 1 training loop.

Acceptance criteria:

- Tiny VAE returns logits, posterior, sampled latents, KL.
- Encoder and decoder both use strictly causal attention.
- Default config matches the paper's structural choices, while tiny config runs fast.

## PR 4: Stage 1 VAE Losses and Pretraining Step

Goal: implement the Stage 1 objective as a reusable training step.

Scope:

- Reconstruction negative log-likelihood.
- KL to base standard normal prior.
- BERT-style mask-loss hook with a simple, configurable masking policy.
- `compute_stage1_vae_loss(...)`.
- Minimal trainer step: forward, loss, backward, optimizer step.
- Diagnostics: reconstruction loss, KL, mask loss, logSNR.

Out of scope:

- Distributed training.
- Large dataset pipeline.

Acceptance criteria:

- A tiny batch can run one Stage 1 optimization step.
- Loss components are returned in a structured object.
- Masking can be disabled with `lambda_mask=0`.

## PR 5: Flow Matching and Noise Schedule Utilities

Goal: implement the latent bridge math independently from the DiT network.

Scope:

- Uniform and LogitNormal timestep sampling.
- Configurable `loc` and `scale/sigma`.
- Linear bridge from clean latent `z0` to base noise `z1`.
- Velocity target and optional `x0` target.
- Flow Matching loss helper.
- Tests for timestep ranges, shapes, and target formulas.

Out of scope:

- Block-causal attention.
- ODE sampler.

Acceptance criteria:

- `sample_timestep` returns values in `(0, 1)` or the configured discrete grid.
- Velocity target test matches the analytical linear bridge.
- `prediction_type` cleanly switches between `velocity` and `x0`.

## PR 6: Block-Causal Packing and Attention Mask

Goal: implement the most architecture-specific part of Cola DLM: packed DiT inputs and block-causal visibility.

Scope:

- Function to build packed DiT sequence:
  - clean context `z0[:, :L-bs]`
  - noisy targets `zt[:, :L]`
  - concatenated length `2L-bs`
- Block ids and segment ids for clean/noisy parts.
- Attention mask matching the paper figure:
  - noisy block `b` sees clean blocks `< b`;
  - noisy block `b` sees itself bidirectionally;
  - no future clean blocks;
  - no other noisy blocks.
- Loss mask for noisy target positions only.
- Stop-gradient handling for historical clean latents.
- Clear tests with a tiny example such as `L=8`, `bs=2`.

Out of scope:

- DiT transformer layers.
- Stage 2 loss integration.

Acceptance criteria:

- Mask tests explicitly verify allowed and denied pairs.
- Packed sequence length equals `2L-bs`.
- The function is documented enough that the training code is straightforward.

## PR 7: Block-Causal Text DiT Backbone

Goal: implement the latent prior network that predicts velocity or denoised latents over packed sequences.

Scope:

- Latent input projection from `latent_dim` to DiT hidden size.
- Time embedding and injection into transformer blocks.
- Optional segment/block embedding if useful.
- DiT transformer stack using PR 2 primitives.
- Output projection to latent dimension.
- Forward API that accepts packed latents and attention mask.
- Tiny shape tests.

Out of scope:

- Full Stage 2 objective.
- CFG and inference.

Acceptance criteria:

- Tiny DiT predicts `[batch, packed_len, latent_dim]`.
- Output can be filtered to noisy target positions for Flow Matching loss.
- Paper-scale config can be instantiated in config only, without running it.

## PR 8: Stage 2 Joint VAE-DiT Objective

Goal: implement joint training with trainable VAE, frozen reference encoder, and DiT Flow Matching prior.

Scope:

- Load/copy frozen reference VAE encoder.
- Compute trainable VAE posterior and reference posterior.
- Compute reconstruction, posterior entropy/log-prob regularizer, mask loss.
- Compute reference KL `KL(q_phi || q_phi_ref)`.
- Build packed DiT inputs and Flow Matching targets.
- `compute_stage2_loss(...)` returning structured components.
- Minimal one-step Stage 2 trainer on a tiny batch.

Out of scope:

- Real checkpoint loading from official weights.
- Distributed training.

Acceptance criteria:

- Reference encoder parameters have `requires_grad=False`.
- Historical clean conditions passed to DiT are detached.
- Tiny Stage 2 step backpropagates into trainable VAE and DiT, not reference encoder.

## PR 9: Block-Wise Inference Sampler

Goal: implement the generation path: prefix encoding, latent block generation, clean condition repaint, and decoding.

Scope:

- Prefix encoding to clean latents.
- Block-wise latent generation from Gaussian noise.
- Euler sampler first; optionally Heun behind a config flag.
- Clean condition repaint for first mixed block.
- CFG hook with conditional/unconditional vector-field combination.
- Decoder call for response logits/tokens.
- KV-cache placeholders where the current simple implementation does not yet optimize cache.

Out of scope:

- High-quality decoding polish.
- Full benchmark evaluation.

Acceptance criteria:

- Tiny model can generate the requested number of latent positions.
- First-block known positions remain exactly fixed under clean condition repaint.
- `num_denoise_steps`, `cfg_scale`, and `block_size` are all config-driven.

## PR 10: Tokenizer, Dataset, and Evaluation Harness

Goal: add the non-training infrastructure needed to compare outputs later.

Scope:

- OLMo 2 tokenizer integration or a documented compatible fallback.
- Simple pretraining dataset interface over tokenized text files.
- Evaluation prompt builders for LAMBADA, MMLU, SIQA, SQuAD, Story Cloze, OBQA, RACE, HellaSwag.
- Normalization and answer matching rules.
- Max new tokens default 32.
- Tiny fake-data tests for prompt formatting and scoring.

Out of scope:

- Downloading all benchmark datasets automatically.
- Reproducing paper-scale scores.

Acceptance criteria:

- Each benchmark has a prompt builder with fixed examples configurable.
- Multiple-choice scoring compares generated option text, not option label.
- LAMBADA and SQuAD have separate normalization paths.

## PR 11: Checkpointing, Training CLI, and Recipes

Goal: make the implementation trainable in small local runs and scalable later.

Scope:

- CLI entrypoints:
  - `train_vae_stage1`
  - `train_stage2`
  - `sample`
  - `evaluate`
- Checkpoint save/load for model, optimizer, scheduler, and config.
- Paper-style config recipes and tiny-debug recipes.
- Mixed precision policy with bf16 autocast and fp32-sensitive ops.
- Basic logging to JSONL or TensorBoard.

Out of scope:

- Multi-node distributed training.
- Full production experiment management.

Acceptance criteria:

- Tiny Stage 1 and Stage 2 runs can resume from checkpoint.
- Config used for a run is stored with the checkpoint.
- Sampling can load a checkpoint and run the PR 9 inference path.

## PR 12: Diagnostics and Reproduction Checks

Goal: add the checks needed to know whether the implementation is behaving like Cola DLM before scaling.

Scope:

- VAE reconstruction accuracy diagnostic.
- VAE logSNR tracking.
- Latent norm and posterior variance tracking.
- DiT Flow Matching loss by block index.
- Attention-mask visualization for small examples.
- First-block condition preservation test.
- Optional latent PCA/UMAP export for RQ2-style analysis.

Out of scope:

- Final paper-scale experimental plots.

Acceptance criteria:

- A debug run produces a compact diagnostics report.
- Mask visualization matches the intended block-causal pattern.
- Reconstruction/logSNR metrics are available during both Stage 1 and Stage 2.

## PR 13: Paper-Scale Configuration Pass

Goal: align the implementation with the documented paper setup after the architecture is stable.

Scope:

- Add paper-scale configs:
  - VAE 4/4 blocks, hidden 1536, FFN 6144, latent dim 16.
  - DiT 24 layers, hidden 2048, FFN 8192, 16 heads.
  - seq len 512, block size 16.
  - AdamW and LR schedule from the paper.
- Estimate parameter counts and compare against reported values.
- Add memory notes and expected hardware constraints.
- Add config variants for ablations: latent dim 64/128, block size 1/64/128, fixed/learnable VAE logSNR, no-BERT loss.

Out of scope:

- Actually running paper-scale training.

Acceptance criteria:

- Parameter count report is generated.
- Reported defaults are traceable back to `00_context.md`.
- All ablation configs inherit from a single clean base config.

## PR 14: Official Release Reconciliation

Goal: once official code/model are released, reconcile this reproduction against them without rewriting the project blindly.

Scope:

- Compare official model shapes, masks, loss parameterization, CFG handling, and noise schedule.
- Record deviations in a compatibility note.
- Add optional adapters only where the official implementation clarifies ambiguous paper details.

Out of scope:

- Replacing clean local code with a wholesale vendor copy.

Acceptance criteria:

- Ambiguities from `00_context.md` are resolved or explicitly kept configurable.
- Any behavior change is covered by tests or a written rationale.

