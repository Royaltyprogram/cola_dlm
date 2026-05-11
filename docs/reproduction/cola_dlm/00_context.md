# Cola DLM Reproduction Context

Sources captured on 2026-05-11:

- Paper HTML: <https://arxiv.org/html/2605.06548v1>
- arXiv source tarball: <https://arxiv.org/e-print/2605.06548v1>
- Project page: <https://hongcanguo.github.io/Cola-DLM/>
- Project page code/model buttons currently point to `https://github.com/ByteDance-Seed/Cola-DLM` and `https://huggingface.co/ByteDance-Seed/Cola-DLM` with a `Soon` badge in the downloaded HTML. No official implementation was available from the page at capture time.

Local source cache:

- `docs/reproduction/cola_dlm/source_cache/arxiv_source/`
- `docs/reproduction/cola_dlm/source_cache/arxiv_html.html`
- `docs/reproduction/cola_dlm/source_cache/project_page.html`

## Reproduction Target

The first implementation target should be an architecture-faithful codebase, not a fully runnable training system. The key is to express the three-part decomposition:

1. Text VAE maps token sequences to continuous per-token latents.
2. Block-causal Text DiT learns a continuous latent prior by Flow Matching.
3. Conditional decoder realizes text from prefix latents and generated response latents.

The default paper configuration for the main scaling comparison is:

- Tokenizer: OLMo 2 tokenizer.
- Vocabulary size: 100,278.
- Sequence length: 512.
- Text latent dimension: 16.
- Text VAE: about 500M parameters, 4 encoder blocks, 4 decoder blocks, hidden size 1,536, FFN size 6,144.
- DiT prior: about 1.8B non-embedding parameters, 24 layers, hidden size 2,048, FFN size 8,192, 16 attention heads, head dimension 128, RoPE.
- Attention pattern: VAE encoder and decoder are strictly causal; DiT is block-causal.
- DiT block size: 16.
- Training noise schedule: LogitNormal timestep sampling with `loc=1` in the main configuration.
- Stage 2 latent-space update: VAE and DiT jointly trained with VAE/DiT learning-rate ratio 1.
- Inference: 16 denoising steps, CFG scale 7.
- Optimization setup: AdamW, peak LR `1.5e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, grad clip `1.0`, 5K linear warmup from `1e-6`, cosine decay to `1e-5`, bf16 autocast with sensitive ops in fp32.
- Batch setup in paper-scale runs: global batch size 1,408, tokens per step 720,896.

## Probabilistic Model

Let `x` be a discrete text sequence and `z0` its continuous latent sequence. The model is:

```text
p(x, z0) = p_theta(x | z0) * p_psi(z0)
p(x)     = integral p_theta(x | z0) * p_psi(z0) dz0
```

The encoder `q_phi(z0 | x)` is an inference network used during training. It is not part of the generative model.

The prior is a continuous normalizing flow / ODE prior:

```text
z1 ~ N(0, I)
dz_t / dt = v_psi(z_t, t)
z0 = Phi_{0<-1}^psi(z1)
```

For sequence modeling, `z0` is partitioned into blocks:

```text
p_psi(z0) = p_psi(z0^(1)) * product_{b=2..B} p_psi(z0^(b) | z0^(<b))
```

This is the architecture-level reason for block-causal DiT.

## Text VAE

The Text VAE establishes a stable text-to-latent interface before learning the final prior.

Expected tensor shape for the default no-compression setup:

```text
tokens: [batch, seq_len]
mu/logvar: [batch, seq_len, latent_dim]
z0: [batch, seq_len, latent_dim]
```

Stage 1 loss:

```text
L_VAE =
  - E_{q_phi(z0|x)} log p_theta(x | z0)
  + beta * KL(q_phi(z0 | x) || p_base(z0))
  + lambda_mask * L_mask
```

Implementation implications:

- `p_base` can be represented as standard normal for the initial VAE interface.
- The encoder outputs posterior `mu` and `logvar`; sampling should use reparameterization.
- VAE logSNR is defined as `log(mean(mu^2) / mean(exp(logvar)))`.
- The paper's default VAE does not compress sequence length. Patch size 1 maps one token to one latent.
- Both encoder and decoder must be strictly causal to prevent future leakage and support streaming generation.
- The figure suggests a clean-token path and a masked-token path. For code, model this as an auxiliary BERT-style masked-token objective over a masked view that encourages semantic latents instead of pure surface copying.

## Stage 2: Joint VAE and Block-Causal DiT

Stage 2 starts from the pretrained Text VAE. The trainable VAE keeps adapting, while a frozen reference VAE encoder from Stage 1 constrains drift.

Stage 2 loss:

```text
L_stage2 =
  lambda_VAE * (
    - E_{q_phi(z0|x)} log p_theta(x | z0)
    + beta * E_{q_phi(z0|x)} log q_phi(z0 | x)
    + lambda_mask * L_mask
  )
  + lambda_fm * L_FM
  + lambda_ref * E_data KL(q_phi(z0|x) || q_phi_ref(z0|x))
```

The reference KL is important. It keeps the latent space from drifting too far while still allowing co-adaptation with the DiT.

## Flow Matching Prior

The strict model learns a prior density, but the practical solver is Flow Matching. With a bridge from clean latent `z0` to noise `z1`:

```text
z0 ~ aggregated posterior
z1 ~ N(0, I)
t  ~ timestep schedule
z_t = (1 - alpha(t)) * z0 + alpha(t) * z1
u_t = d z_t / dt
```

Formal FM loss:

```text
L_FM = sum_b E || v_psi(z_t^(b), t; z0^(<b)) - u_t^(b) ||_2^2
```

The paper text formalizes a vector-field target, while the workflow figure labels the module output as a denoised latent with MSE to the clean latent. For faithful code, implement the DiT as a configurable head:

- `prediction_type="velocity"` for the formal Flow Matching objective.
- `prediction_type="x0"` if reproducing the figure's denoised-latent parameterization.

The architecture and attention mask are the same in both cases.

## Block-Causal DiT Attention

For a target block `b`, the visible set is:

```text
V_b = { stop_gradient(z0^(<b)), z_t^(b) }
```

This means:

- Historical blocks are clean latents and are treated as conditions.
- The current block is noisy and is the denoising target.
- Attention is bidirectional within the current noisy block.
- Attention is causal across blocks.
- Gradients from the DiT prior should not flow into historical clean condition latents.

The workflow figure packs all block prediction problems into one sequence for efficient training. For latent sequence length `L` and block size `bs`, it concatenates:

```text
clean_context = z0[:, : L - bs]        # clean blocks 1..B-1
noisy_targets = z_t[:, : L]            # noisy blocks 1..B
dit_input = concat(clean_context, noisy_targets)
packed_len = 2 * L - bs
num_packed_blocks = packed_len / bs
```

For each query in noisy target block `b`, allow attention to:

- clean context blocks `< b`
- noisy target block `b`

Do not allow it to see future clean blocks or other noisy blocks. Compute FM loss only on noisy target positions.

## Noise Schedule

The paper studies timestep shift through LogitNormal sampling:

```text
u ~ Normal(mu, sigma^2)
s = sigmoid(u)
t = T * s
```

`mu` is the `loc` shift. Larger `loc` shifts sampling mass toward later timesteps. `sigma` controls spread.

Implementation note: the main result repeatedly specifies `loc=1`. One discussion table mentions `scale=0`, which is not a valid LogitNormal standard deviation if interpreted literally. Treat `scale` as unresolved until official code clarifies it; expose it as a config field.

## Inference

Prefix-conditioned generation:

1. Encode prefix tokens into clean prefix latents with the Text VAE encoder.
2. Generate response latents block by block. For each block, sample `epsilon ~ N(0, I)` and integrate the DiT vector field from `t=1` to `t=0` conditioned on prefix and previous generated clean latents.
3. Decode the response with the Text VAE decoder conditioned on prefix latents and generated response latents.

The paper reports that most gains appear by about 8-10 denoising steps and saturate around 16-32 steps. Main comparison uses 16 steps and CFG 7.

First generation block handling is important. If the prefix ends inside a block, the first generated block contains both known prefix latents and unknown response latents. The best reported strategy is clean condition repaint:

- keep known prefix-region latents fixed and clean for the whole denoising trajectory;
- only transport the unknown region;
- use the known region as a stable boundary condition.

Partial repaint and left/right padding are weaker in the reported ablation.

## Evaluation Protocol

The main comparison avoids perplexity as the primary metric because likelihood-oriented estimates can be misaligned with latent generation quality. All models are evaluated through unified few-shot generation:

- MMLU, RACE, OBQA, HellaSwag: 2-shot multiple choice, generate option text, not option label.
- Story Cloze: 2-shot story ending, anchor `End:`.
- SIQA: 2-shot, 3 options.
- LAMBADA: 0-shot direct continuation, first generated word is the prediction.
- SQuAD: 1-shot extractive QA generation, normalized exact match.
- Max evaluation new tokens: 32.

## Open Implementation Questions

These are not fully specified by the paper/page and should stay configurable in code:

- Exact masking policy and weighting for the BERT-style `L_mask`.
- Exact values of `beta`, `lambda_mask`, `lambda_VAE`, `lambda_fm`, and `lambda_ref`.
- Exact CFG training dropout / unconditional-condition construction.
- Exact LogitNormal `scale` used in the main configuration.
- Whether the released implementation uses velocity prediction or denoised-latent prediction internally.
- Exact external pretraining corpus mixture; the paper says open-source pretraining data but does not name a complete mixture in the captured text.

## Recommended Code Modules

Keep the initial implementation simple:

- `configs.py`: dataclasses for VAE, DiT, diffusion, training, inference.
- `vae.py`: causal Text VAE encoder/decoder, posterior object, reparameterization, KL/logSNR utilities.
- `dit.py`: transformer blocks, time embedding, latent input projection, output head.
- `block_causal_mask.py`: packed DiT sequence construction and attention mask.
- `flow_matching.py`: timestep sampling, bridge construction, velocity/x0 targets.
- `losses.py`: Stage 1 and Stage 2 losses.
- `sampling.py`: block-wise ODE sampler, clean condition repaint, CFG hook.
- `evaluation_notes.md`: prompt templates and matching rules after architecture code exists.

