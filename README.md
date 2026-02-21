I came across this interesting idea (earlier discussed on reddit) and wanted to try it and conduct my own experiments and/or improvements on top of it.

### Notes:

- I've added Blockwise Quantization - So instead of computing one alpha for the entire layer, we divide the flattened weight matrix into blocks of 256 elements and compute a separate alpha for each block. Each block thus has its own scaling factor adapted to its local weight distribution. A block with small-magnitude weights gets a small alpha, preserving precision in that region.
- `RMSNorm` is defined before `BitLinear` to normalizes the input before the linear transformation. Pre-norm (old approach) normalizes at block level, but the data may drift before reaching `BitLinear`, so normalizes right before quantization make sure activations are in the correct scale for ternary weight operations (random discord anon).
- Progressive `SEQ_LEN` training is still buggy but its was implemented i saw it somewhere that learning short range dependencies before long range is beneficial. The model first learns local patterns, word-level statistics, common bigrams and trigrams, etc. Then, when we increase the sequence length, it can build on this foundation to learn longer-range dependencies.
- trainv4.py still uses `GatedConvMixer` for now - people new to this, instead of maintaining and updating a recurrent state (in `GatedDeltaNet`), it applies a depthwise 1D convolution across the sequence. The input is projected to twice its dimension, split into a gate and a value, the value is convolved with a kernel of size 8, and the result is modulated by the sigmoid-activated gate before being projected back down.
- Evals are just 💀 for current run. Previous run (shown below) was better without per block alpha amd rmsnorm before linear. Maybe, i'll remove norm from BitLinear, keep blockwise quantization only.
- Next, my idea is to replace `GatedConvMixer` with `GatedDeltaNet`, which maintains a recurrent state matrix S that is updated at each timestep using the gated delta rule: the model computes what it expects to know about the current key (Sk), calculates the delta between the actual value and this expectation (v_t - Sk), then updates the state by decaying the old information (via gating) and adding the precise correction (via the delta rule).
- The thing with `GatedDeltaNet` (and if im not wrong Kimi Attn) is it enables longer sequences, 1024/2048/etc, and maintains O(N) complexity! But it would be slower due to the sequential loop overhead.
- Added `GatedDeltaNet` - it's naive, but combining GatedDeltaNet with ternary quantization is interesting. The implementation doesn't parallelize across sequence length, whereas in nvlabs repo we can see they've used chunkwise parallel algo. For output norm nvlabs uses FusedRMSNormSwishGate, and mine applies output gate via g_proj but no norm.
- Im not working with large seq len so i can skip Flash attn style chunking.  ~~I just need to see how i can implement parallel computation within chunk - we need to compute delta updates in parallel. lol <think> time to use opencode. </think>~~. Anyway, I won't test things for seq len > 1024 so I can skip parallel chunking for now.


### Training run before adding blockwise quantisation:
```
Parameters: ~5m
d_model: 192
Blocks: 6
GLU hidden dim: 512
Blocks = 6
Sequence length: 256
Vocab size: 10k
Weight tying
Total tokens trained: 40.4M
Best validation loss: 2.003
```

### Eval
<img width="900" height="480" alt="screenshot-2026-02-21_11-45-08" src="https://github.com/user-attachments/assets/0e941944-0eee-4499-b2b1-8ae3d83b4cdb" />



### Sample:
```
The little girl was very sad. She tried to go away but the cop said no.

Once upon a time, there was a little girl named Lily. She loved to play outside in her backyard. One day, she found a shiny toy, and accidentally knocked it down. It was very expensive and had a loud noise.

Lily felt sad and upset. She wanted to go back to her mom. She wanted to keep going. She was scared and her mom told her that she had to go to the hospital.
```

### Internal monologue

- Per-layer scaling: Each layer has its own alpha (mean absolute value) for dynamic range.
- Linear scaling with sequence length instead of quadratic.
- Gating Mechanism.
- Up-projects to 2×dim, splits into gate and value.
- Gate uses sigmoid activation for [0,1] range.
- Value undergoes causal depthwise convolution.
- Element-wise multiplication then down-projection.
-------------------------------------------------------------------
# FlashLM

A family of ternary (1.58-bit) language models that train and run entirely on CPU. Weights are constrained to {-1, 0, +1}, so inference uses only additions and subtractions — no floating-point multiplies.

## Models

| Version | Params | Dataset | Val Loss | BPC | Status |
|---------|--------|---------|----------|-----|--------|
| **v4 "Bolt"** | 4.3M | TinyStories | 2.10 | 0.88 | ✅ Current |
| v3 | 13.6M | FineWeb-Edu | 6.80 | — | Archived |

## v4 "Bolt" Architecture

```
Embedding (10K × 192, float, weight-tied)
  → 6 × BoltBlock:
      RMSNorm → Gated Causal DepthwiseConv (k=8) → residual
      RMSNorm → Ternary GLU (SiLU, 192→512→192) → residual
  → RMSNorm → Output Head (tied to embedding)
```

All linear layers inside BoltBlocks use ternary BitLinear quantisation (straight-through estimator, α = mean|W|). The only floating-point operations are the embedding lookup, RMSNorm, and the tied output projection.

## Quick Start

### Training

```bash
# Install dependencies
pip install torch tiktoken datasets

# Auto-detect hardware and train
python train.py

# Small model (4.3M params), 2-hour run
python train.py --small --hours 2

# Large model (15.7M params), train until convergence
python train.py --large

# Resume from checkpoint
python train.py --resume checkpoints/flashlm_v4_step_5000.pt

# Custom configuration
python train.py --large --batch 64 --lr 2e-3 --epochs 5
```

The script auto-detects CPU cores and RAM, selects an appropriate model size, downloads TinyStories, builds a frequency-based 10K vocabulary (99.9% coverage), caches tokenized data to disk, and begins training. Subsequent runs skip download and tokenization.

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--small` | Force v4-small (d=192, 6 blocks, ~4.3M params) | Auto |
| `--large` | Force v4-large (d=384, 8 blocks, ~15.7M params) | Auto |
| `--resume PATH` | Resume training from a checkpoint | — |
| `--epochs N` | Maximum epochs | 10 |
| `--hours H` | Wall-clock time limit | None |
| `--batch N` | Batch size | Auto |
| `--lr FLOAT` | Peak learning rate | Auto |
| `--seed N` | Random seed | 42 |
| `--save-dir PATH` | Checkpoint directory | `checkpoints/` |

## Files

| File | Description |
|------|-------------|
| `train.py` | Standalone training script for FlashLM v4 |
| `eval_bpc.py` | BPC evaluation script (FlashLM v4 vs TinyStories-1M) |
| `FlashLMv3.ipynb` | Original v3 notebook (archived) |


## Results

**FlashLM v4 vs TinyStories-1M** (500 validation stories):

| Metric | FlashLM v4 | TinyStories-1M |
|--------|-----------|----------------|
| Params | 4.3M (ternary) | 3.7M (float32) |
| BPC | 0.88 | 0.62 |
| PPL | 15.05 | 6.72 |
| Hardware | 2-thread CPU | V100 GPU |
| Tokens seen | 10.6M | ~470M |
| Training time | 2 hours | Hours (GPU) |

The BPC gap is primarily due to undertraining — v4 has seen only 2.3% of the data the baseline used, and loss was still decreasing when the 2-hour time limit was reached.

## Links

- **Model & Weights:** [HuggingFace](https://huggingface.co/changcheng967/flashlm-v4-bolt)
- **Demo:** [HuggingFace Space](https://huggingface.co/spaces/changcheng967/flashlm-v4-demo)
- **v3 Model:** [HuggingFace](https://huggingface.co/changcheng967/flashlm-v3-13m)
- **v3 Demo:** [HuggingFace Space](https://huggingface.co/spaces/changcheng967/flashlm-v3-demo)

## Roadmap

- [ ] Extended training on Ryzen 7950X3D (16 cores, 128GB RAM)
- [ ] Scale to ~15M params (v4-large)
- [ ] Curriculum learning (TinyStories → SimpleStories → filtered FineWeb-Edu)
- [ ] ONNX / C inference runtime
- [ ] BPC evaluation script (`eval_bpc.py`)

## Inspired By

- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764)
- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)
- [TinyStories](https://arxiv.org/abs/2305.07759)

## License

MIT — see [LICENSE](LICENSE).
