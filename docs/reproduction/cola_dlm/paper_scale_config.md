# Paper-Scale Config Notes

The paper-scale recipes in this repo are intended for configuration review,
reporting, and parameter-count auditing. They are not local training recipes.
Use the tiny-debug recipes for local smoke tests.

## Paper Defaults

| Section | Paper-scale default | Source |
| --- | --- | --- |
| Tokenizer | OLMo 2 tokenizer; vocabulary size `100,278`. | [`00_context.md`, Reproduction Target](00_context.md#reproduction-target) |
| Sequence length | `512` tokens. The VAE patch size is `1`, so the default latent length also stays `512`. | [`00_context.md`, Reproduction Target](00_context.md#reproduction-target) and [`Text VAE`](00_context.md#text-vae) |
| Text VAE | Hidden size `1,536`; FFN size `6,144`; `4` encoder blocks; `4` decoder blocks; latent dimension `16`; strictly causal attention. | [`00_context.md`, Reproduction Target](00_context.md#reproduction-target) and [`Text VAE`](00_context.md#text-vae) |
| DiT prior | Hidden size `2,048`; FFN size `8,192`; `24` layers; `16` heads; head dimension `128`; block size `16`; RoPE; block-causal attention. | [`00_context.md`, Reproduction Target](00_context.md#reproduction-target) and [`Block-Causal DiT Attention`](00_context.md#block-causal-dit-attention) |
| Flow matching | Velocity prediction with LogitNormal timestep sampling at `loc=1`; `scale` remains unresolved in the captured sources and is kept configurable. | [`00_context.md`, Flow Matching Prior](00_context.md#flow-matching-prior) and [`Noise Schedule`](00_context.md#noise-schedule) |
| Optimizer | AdamW; peak LR `1.5e-4`; `5,000` linear warmup steps from `1e-6`; cosine decay to min LR `1e-5`; weight decay `0.01`; betas `(0.9, 0.95)`; grad clip `1.0`; bf16 autocast with sensitive ops in fp32. | [`00_context.md`, Reproduction Target](00_context.md#reproduction-target) |
| Batch scale | Global batch size `1,408`; tokens per step `720,896`. | [`00_context.md`, Reproduction Target](00_context.md#reproduction-target) |

Stable entry points:

- `configs/stage1_paper.json`
- `configs/stage2_paper.json`

The generated parameter-count report is in
[`parameter_counts.md`](parameter_counts.md).

## Memory Notes

bf16 parameter storage alone is not the dominant planning cost for paper-scale
training. AdamW adds optimizer moment tensors, gradients add another copy of
trainable parameters, and mixed-precision training may keep selected values in
fp32 depending on the training path.

Activation memory grows with sequence length, the packed DiT length, hidden
size, layer count, and batch or microbatch size. For the default DiT packing,
`L=512` and `block_size=16` produce a packed length of
`2L - block_size = 1,008` positions before accounting for hidden activations
across `24` transformer layers.

The paper global batch size also implies gradient accumulation or data
parallelism when the per-device microbatch is smaller than `1,408`. Combined
with AdamW state and activation storage, these recipes should be treated as a
multi-accelerator training concern rather than a single-machine smoke-test
target.

For local verification, use the tiny-debug recipes:

- `configs/stage1_tiny_debug.json`
- `configs/stage2_tiny_debug.json`
- `configs/inference_tiny_debug.json`

