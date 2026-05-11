# Cola DLM Parameter Counts

Generated from checked-in paper-scale recipes. The CLI initializes models on the PyTorch `meta` device by default so this report can be produced without allocating paper-scale tensors.

## Resolved Configs

- Stage 1 VAE config: `configs/stage1_paper.json`
- Stage 2 DiT config: `configs/stage2_paper.json`
- VAE initialization device: `meta`
- DiT initialization device: `meta`

## Summary

| Component | Local count | Paper reported value | Accounting note |
| --- | ---: | --- | --- |
| Text VAE total | 688,783,904 (688.8M) | about 500M | Includes local decoder output projection. |
| Text VAE token embeddings | 308,054,016 (308.1M) | 308,054,016 in the appendix | Counts encoder and decoder token embeddings only. |
| Text VAE decoder output projection | 154,027,008 (154.0M) | not listed in appendix embedding row | Counted with the local decoder and VAE total. |
| DiT total | 1,216,962,576 (1217.0M) | about 1.8B | No `nn.Embedding` parameters are enabled in the paper config. |
| DiT non-embedding backbone | 1,216,962,576 (1217.0M) | about 1.8B | Local transformer block is smaller than the reported paper scale. |

## VAE Components

| Component | Trainable | Non-trainable | Total |
| --- | ---: | ---: | ---: |
| Encoder | 267,389,984 | 0 | 267,389,984 (267.4M) |
| Decoder | 421,393,920 | 0 | 421,393,920 (421.4M) |
| Total | 688,783,904 | 0 | 688,783,904 (688.8M) |

| VAE accounting bucket | Parameters |
| --- | ---: |
| Encoder token embedding | 154,027,008 (154.0M) |
| Decoder token embedding | 154,027,008 (154.0M) |
| Decoder output projection | 154,027,008 (154.0M) |
| All token embeddings | 308,054,016 (308.1M) |
| Non-embedding VAE parameters | 380,729,888 (380.7M) |

## DiT Components

| Component | Parameters |
| --- | ---: |
| Input projection | 34,816 (0.0M) |
| Timestep embedding MLP | 8,392,704 (8.4M) |
| Transformer layers | 1,208,500,224 (1208.5M) |
| Segment embedding | 0 (0.0M) |
| Output norm and projection | 34,832 (0.0M) |
| All embeddings | 0 (0.0M) |
| Non-embedding DiT parameters | 1,216,962,576 (1217.0M) |
| Total DiT parameters | 1,216,962,576 (1217.0M) |

## Implementation Notes

- The paper appendix reports Cola DLM embedding parameters as `308,054,016` and AR/LLaDA embedding parameters as `410,738,688`.
- The local VAE has separate encoder and decoder token embeddings. Their sum matches the appendix Cola DLM embedding row.
- The local decoder output projection is an untied vocabulary projection. It is not counted with embeddings here; it is counted with the decoder and with total VAE parameters.
- The local DiT count is lower than the paper's `about 1.8B` non-embedding backbone report. This report records the current checked-in implementation instead of adjusting architecture constants to force a match.
