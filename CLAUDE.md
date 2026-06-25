# MathNano — CLAUDE.md

This file is read by Claude Code at the start of every session.
It defines what this project is, every architectural decision and why it was made,
how code should be written, and what the educational goals are.

---

## ⚠️ Reality Check (read `experiments/nanochat_reading_notes.md` first)

The original draft of this file described nanochat from memory and got several things
**wrong**. The notes file is the verified-from-source source of truth. Key deltas that
override anything below:

- **nanochat is a modded-nanoGPT network, not the vanilla LLaMA stack.** It uses **relu²**
  MLP (not SwiGLU), **untied** embed/lm_head (not weight-tied), **parameter-free** RMSNorm,
  **QK-norm**, value embeddings, smear/backout, and sliding-window attention.
- **Attention is full MHA by default** (depth-scaling sets `n_kv_head = n_head`); GQA is
  supported but unused. The "n_kv_heads=2" claim below is aspirational, not what runs.
- **Vocab = 32,768** (init loss ≈ ln(32768) ≈ **10.4**); the primary metric is **bits-per-byte**.
- **No uint16 `.bin` pipeline.** Pretraining reads **parquet shards** (`text` column) from
  `$NANOCHAT_BASE_DIR/base_data_climbmix/`; the tokenizer trains from the same text. We feed
  MathPile by dropping parquet shards there — **zero nanochat edits**.
- **Real scripts**: `tok_train.py`, `base_train.py`, `chat_sft.py`, `chat_rl.py` (GRPO is
  hardcoded to GSM8K). CLI flags use **dashes** (`--device-batch-size`, …); there is no
  `--data_path` / `--hf_repo`.
- **Hardware**: nanochat targets 8×H100. On our RTX 4090 (Ada) it uses the SDPA fallback (FA3
  is Hopper-only); `--fp8` is unavailable.

The architecture explanations below are still useful as *"what the canonical modern stack
does and why"* — just don't assume they describe nanochat's exact code. See
`ARCHITECTURE.md §13` for the precise differences.

### Project shape (current)
Two tracks + a product (see `PROJECT_PLAN.md`): **Track A** = nanochat ~200M from scratch
(the deep-learning artifact); **Track B** = SFT+GRPO on a small pretrained base, Qwen2.5
(the shippable, genuinely capable model); **Product** = a chat web UI + clean inference API
(`serve/`). Compute: single RTX 4090 spot. Budget ceiling £25 is tight — **if it runs low,
finish Track B (the product) first; Track A pretrain can be shortened.**

---

## What This Project Is

A mathematical reasoning language model trained **from scratch** on competition and
word-problem mathematics. It uses Karpathy's nanochat (released October 2025) as the
base codebase, modified for the math domain.

The model is nanochat depth=16 (~200M parameters). It trains in 10–25 hours on a
single RTX 4090 (~£6–10 on RunPod spot pricing). Total compute budget: £25.

This is an **educational project first**. The goal is not just a working model — it
is to understand every architectural and training decision deeply. When writing or
modifying code, always explain:
1. What the code does mechanically
2. Why this approach was chosen (not just a description of what it does)
3. What would break or degrade if this component were removed or changed

---

## The Three-Stage Training Pipeline

```
STAGE 1: PRETRAIN           STAGE 2: SFT                 STAGE 3: GRPO
────────────────────        ─────────────────────        ─────────────────────
Dataset: MathPile           Dataset: OpenMathInstruct-2  Dataset: GSM8K + MATH
9.5B tokens                 + GSM8K + MATH               (problems w/ answers)

Objective:                  Objective:                   Objective:
next-token prediction       instruction following        RL with verifiable
on raw math text            with chain-of-thought        binary rewards

Model learns:               Model learns:                Model learns:
math notation, proof        how to format and            to get correct answers
structure, equation         present a step-by-step       not just mimic format
patterns, jargon            solution

Output:                     Output:                      Output:
a model that "speaks        a chatbot that follows       a reasoning model
math" as autocomplete       instructions                 that improves through RL
```

Stage 3 (GRPO) is the same mechanism that made DeepSeek-R1 impressive, at 1/50th the
scale. The reward is binary: extract the final answer from the model's output, compare
to ground truth. Correct = +1. Wrong = -1. No human labelling. No reward model. Math
provides this for free.

---

## Architecture Decisions

These decisions are **fixed for this project**. Do not change them without explaining
the architectural trade-off. All of them are based on the converged "LLaMA stack" —
the design that Llama 4, Mistral, Gemma, Qwen, and Phi all independently arrived at.

### 1. RoPE — Rotary Positional Embeddings

**Replaces**: learned absolute embeddings (GPT-2), sinusoidal (original transformer),
ALiBi (Bloom).

**Why RoPE**: Position is encoded as a rotation matrix applied to the Query and Key
vectors inside attention. For token at position m, each pair of dimensions (2i, 2i+1)
in the head is rotated by angle m * θ_i, where θ_i = 10000^(-2i/d_head).

The rotation has a crucial property: when you compute the dot product q_m · k_n, the
result depends only on the *relative position* (m - n), not on absolute positions.
This means the model generalises to sequence lengths it has never seen in training.

Lower dimension indices rotate fast (high frequency, sensitive to nearby tokens).
Higher dimension indices rotate slow (low frequency, sensitive to long-range patterns).
This gives the model a built-in sense of scale.

**The code pattern**:
```python
# cos and sin tables for each position and each dimension pair
# cos_cached: (max_seq_len, head_dim // 2)
# sin_cached: (max_seq_len, head_dim // 2)

def apply_rope(x, cos, sin, position_ids):
    # x: (batch, n_heads, seq_len, head_dim)
    # Split into even/odd dimension pairs
    x_even = x[..., 0::2]   # (batch, n_heads, seq_len, head_dim//2)
    x_odd  = x[..., 1::2]   # (batch, n_heads, seq_len, head_dim//2)
    # Apply rotation: [x_even, x_odd] * [[cos, -sin], [sin, cos]]
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd  = x_even * sin + x_odd * cos
    # Interleave back: (batch, n_heads, seq_len, head_dim)
    return torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)
```

### 2. RMSNorm — Root Mean Square Normalisation

**Replaces**: LayerNorm (original transformer and GPT-2).

**Why RMSNorm**: LayerNorm computes mean μ and variance σ², subtracts μ, divides by σ²,
then applies a learnable scale γ and bias β. RMSNorm skips the mean computation entirely
— it only divides by the root mean square:

  RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ

Empirically identical quality to LayerNorm, ~10% faster. The re-centering (subtracting
mean) turns out to be unnecessary.

**Pre-norm placement**: We place normalisation *before* the attention and FFN layers,
not after (which the original transformer did). Pre-norm gives healthier gradients in
deep networks — the residual path stays clean.

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # Learnable scale per dimension. No bias — RMSNorm doesn't need it.
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # Compute RMS across the last dimension
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        # Normalise and scale
        return (x / rms) * self.weight
```

### 3. SwiGLU — Swish-Gated Linear Unit

**Replaces**: ReLU (original transformer), GeLU (GPT-2/BERT).

**Why SwiGLU**: The feed-forward network in a transformer is usually:
  FFN(x) = activation(x W₁) W₂

SwiGLU changes this to a gated formulation:
  FFN(x) = (Swish(x W₁) ⊙ (x W₂)) W₃

The ⊙ is elementwise multiplication. It acts as a learned gate — the network decides
which activations to let through, per feature, per position. This consistently
outperforms ungated activations at the same compute budget.

**Important dimension change**: SwiGLU has 3 weight matrices instead of 2. To keep the
parameter count equal to a standard FFN, reduce the hidden dimension by 2/3:
- Standard FFN: d_model → 4*d_model → d_model (2 matrices)
- SwiGLU FFN: d_model → (8/3)*d_model → d_model (3 matrices, ~same params)

Swish(x) = x * sigmoid(x). It's like ReLU but smooth and slightly negative for x < 0.

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Hidden dim is 8/3 of d_model, rounded to nearest multiple of 64 for hardware
        hidden = int(d_model * 8 / 3)
        hidden = (hidden + 63) // 64 * 64
        # Three projections, all without bias (modern practice)
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate path
        self.w2 = nn.Linear(d_model, hidden, bias=False)  # value path
        self.w3 = nn.Linear(hidden, d_model, bias=False)  # projection back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # Gate: Swish applied to w1, elementwise multiply with w2
        # WHY: the gate learns which features to amplify or suppress
        gate = F.silu(self.w1(x))   # silu IS swish: x * sigmoid(x)
        value = self.w2(x)
        # (batch, seq_len, hidden)
        return self.w3(gate * value)
        # (batch, seq_len, d_model)
```

### 4. GQA — Grouped Query Attention

**Replaces**: Multi-Head Attention (MHA, original transformer).

**Why GQA**: During autoregressive generation, every previously computed Key and Value
tensor must be stored in the KV cache. With standard MHA and n_heads attention heads,
the cache grows as: n_heads × seq_len × head_dim × 2 (K and V). This is the primary
memory bottleneck for long generations.

GQA groups multiple Query heads to share a single K/V head pair. With n_heads=8 Q
heads and n_kv_heads=2 K/V heads, 4 Q heads share each K/V head. The cache is 4x
smaller. Empirically, quality loss is negligible.

nanochat default: n_heads=8, n_kv_heads=2.

```python
def grouped_query_attention(q, k, v, n_heads, n_kv_heads):
    # q: (batch, n_heads,    seq_len, head_dim)
    # k: (batch, n_kv_heads, seq_len, head_dim)  ← fewer K heads than Q heads
    # v: (batch, n_kv_heads, seq_len, head_dim)

    groups = n_heads // n_kv_heads   # = 4 in our case

    # Expand K and V so each Q head has a corresponding K/V head
    # repeat_interleave copies [k0, k1] to [k0, k0, k0, k0, k1, k1, k1, k1]
    k_expanded = k.repeat_interleave(groups, dim=1)
    # k_expanded: (batch, n_heads, seq_len, head_dim)
    v_expanded = v.repeat_interleave(groups, dim=1)

    # Standard scaled dot-product attention
    scale = q.shape[-1] ** -0.5
    scores = (q @ k_expanded.transpose(-2, -1)) * scale
    # scores: (batch, n_heads, seq_len, seq_len)

    # Causal mask: token at position t can only attend to positions 0..t
    # WHY: this is a decoder-only model. Future tokens don't exist at inference time.
    mask = torch.triu(torch.ones(scores.shape[-2:]), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    # attn: (batch, n_heads, seq_len, seq_len)

    output = attn @ v_expanded
    # output: (batch, n_heads, seq_len, head_dim)
    return output
```

### 5. No Bias Terms

All `nn.Linear` layers use `bias=False`. This is the modern default across all major
LLMs. Cleaner gradients, fewer parameters, no empirical quality loss. The model learns
to represent offsets through normalisation and the residual stream.

### 6. Muon Optimizer (for 2D parameters)

**Replaces**: AdamW alone.

**Why Muon**: Muon orthogonalises gradient updates for 2D weight matrices using
Newton-Schulz iteration. The gradient for a weight matrix W is typically correlated
across rows and columns. Orthogonalisation decorrelates it, leading to faster and more
stable convergence, especially in early training.

nanochat uses:
- Muon for all 2D parameters (weight matrices)
- AdamW for all 1D parameters (embedding table, RMSNorm scales)

You do not need to implement Muon — nanochat ships with it. But you should understand
that it exists and why it's better than plain AdamW.

---

## nanochat's --depth Parameter

nanochat auto-scales all architecture dimensions from a single integer using
Chinchilla-optimal scaling laws. You set depth and everything else follows.

| depth | approx params | d_model | n_heads | n_kv_heads | n_layers | use              |
|-------|---------------|---------|---------|------------|----------|------------------|
| 8     | ~60M          | 512     | 8       | 2          | 8        | debug / smoke    |
| 16    | ~200M         | 1024    | 8       | 2          | 16       | our main model   |
| 20    | ~560M         | 1280    | 10      | 2          | 20       | out of budget    |

**Rule**: always test with depth=8 first. A full depth=8 run costs ~£0.50–1 and will
catch data pipeline bugs, config mistakes, and OOM errors before you commit to depth=16.

---

## Repository Structure

```
mathnano/
├── CLAUDE.md               ← this file (read first)
├── PROJECT_PLAN.md         ← phases, milestones, budget breakdown
├── ARCHITECTURE.md         ← every component explained from first principles
├── DATASETS.md             ← download commands, processing pipeline, formats
├── RUNPOD.md               ← pod setup, checkpointing, cost management
│
├── nanochat/               ← cloned nanochat repo (base codebase, do not edit)
│
├── mathnano/               ← our additions and modifications
│   ├── data/
│   │   ├── prepare_mathpile.py     ← tokenise MathPile for nanochat
│   │   ├── prepare_sft.py          ← format OpenMathInstruct + GSM8K + MATH
│   │   └── inspect_data.py         ← visualise samples, token statistics
│   ├── rewards/
│   │   └── math_reward.py          ← GRPO reward: extract answer, check correctness
│   ├── eval/
│   │   ├── eval_gsm8k.py           ← evaluate on GSM8K test set
│   │   └── eval_math.py            ← evaluate on MATH benchmark
│   └── config/
│       ├── pretrain_d8.sh          ← smoke test config
│       └── pretrain_d16.sh         ← main training config
│
├── notebooks/
│   ├── 01_tokenizer_deep_dive.ipynb
│   ├── 02_attention_from_scratch.ipynb
│   └── 03_training_curves.ipynb
│
├── track_b/                ← capable product model: Qwen2.5 SFT + GRPO (TRL/PEFT stack)
│   ├── sft.py
│   ├── grpo.py
│   └── configs/
│
├── serve/                  ← PRODUCT: FastAPI inference API + web chat UI + Dockerfile
│   ├── api.py
│   ├── inference.py
│   ├── ui/
│   └── tests/
│
└── experiments/
    ├── nanochat_reading_notes.md   ← Phase 0 notes (verified-from-source: read first)
    └── logs/                       ← training run logs
```

Note: `mathnano/rewards/math_reward.py` is the **shared** verifiable reward used by Track B
GRPO and the eval harness (one source of truth for correctness). Track A reuses nanochat's
built-in GSM8K reward in `tasks/gsm8k.py` and an added `tasks/math.py` for MATH.

---

## Code Quality Standards

### Tensor shape annotations (mandatory)

Every tensor used in attention, normalisation, or matrix operations must have its shape
in a comment. This is non-negotiable — it is how you understand what is happening.

```python
# q: (batch_size, n_heads, seq_len, head_dim)
# k: (batch_size, n_kv_heads, seq_len, head_dim)
# v: (batch_size, n_kv_heads, seq_len, head_dim)
# Note: n_kv_heads < n_heads in GQA. k and v have fewer "copies" than q.
```

After any reshape, permute, or transpose, add a comment showing the new shape.

### The WHY comment pattern

Every non-trivial operation gets two lines:
1. The mechanical description (what it does)
2. The architectural reason (why this way, not another way)

```python
# Scale attention scores by 1/sqrt(head_dim).
# WHY: without scaling, dot products grow in magnitude as head_dim increases.
#      Large dot products → saturated softmax → near-zero gradients → dead attention.
#      Scaling keeps variance ~1 regardless of head_dim.
scale = head_dim ** -0.5
scores = (q @ k.transpose(-2, -1)) * scale
```

### Shape assertions in new modules

Add assertions when building any new module from scratch:
```python
assert x.shape[-1] == self.d_model, \
    f"Expected d_model={self.d_model}, got {x.shape[-1]}"
```

### Test before training

Any new module (reward function, data processor, eval script) must be tested on a
tiny synthetic input before use in training. Do not discover bugs mid-run on RunPod.

---

## What To Explain When The User Asks "What Does This Do"

Give four things, in order:

1. **One sentence**: the core function.
2. **Intuition**: why does it work this way? What problem does it solve?
3. **Code tour**: walk through the relevant lines, with tensor shapes.
4. **What breaks if removed**: what would happen to the model without this component.

Example for layer normalisation:
1. "It normalises the activations at each position to have unit scale."
2. "Without normalisation, activations can grow unboundedly as they pass through layers,
   causing exploding/vanishing gradients. Normalisation keeps the distribution stable
   so the same learning rates work throughout training."
3. "self.weight (γ) is a learnable scale per dimension. We compute RMS of each
   position's activation vector, divide by it, then multiply by γ."
4. "Without it, deep networks become untrainable — loss oscillates or diverges after
   a few thousand steps."

---

## Domain: Word Problem and Competition Mathematics

We chose word problems (not symbolic algebra) because:

1. **GRPO requires a verifiable scalar reward**. Word problems have numerical answers
   you can extract and compare. Symbolic algebra requires a symbolic evaluator that is
   much harder to implement.

2. **Reasoning chains are readable**. You can look at the model's output and understand
   what it's doing. This is how you learn from watching the model improve.

3. **Established benchmarks exist**: GSM8K (8500 grade-school problems), MATH (12500
   competition problems at 5 difficulty levels), AMC (competition problems). These
   provide clear signals for progress.

4. **MathPile is sized correctly**: 9.5B tokens matches nanochat's depth=16 optimal
   training budget. No awkward truncation or padding required.

---

## Evaluation Targets

These are realistic targets for a 200M model trained for <£30. They are not impressive
by frontier standards but are meaningful at this scale.

| Benchmark   | Pretrain only | After SFT   | After GRPO  | Notes                    |
|-------------|---------------|-------------|-------------|--------------------------|
| GSM8K       | ~5%           | ~20–30%     | ~35–45%     | Grade school word problems |
| MATH        | ~1%           | ~5–10%      | ~10–18%     | Competition problems      |
| AMC         | ~0%           | ~2–5%       | ~5–10%      | Hard competition          |

The GRPO improvement over SFT baseline (+5–15pp) is the most important signal in this
project. It demonstrates that RL with verifiable rewards genuinely teaches the model
to solve problems, not just format solutions.

---

## Budget

Total: £25 (~$32).
GPU: RTX 4090 spot on RunPod (~$0.34/hr). Always use spot for training.

| Run                          | Hours    | Cost      |
|------------------------------|----------|-----------|
| Smoke tests (depth=8, ×2–3)  | 2–4 hr   | ~£1–2     |
| Pretraining depth=16         | 15–25 hr | ~£6–10    |
| SFT                          | 3–6 hr   | ~£1–2.50  |
| GRPO                         | 6–12 hr  | ~£2.50–5  |
| Evaluation                   | 1–2 hr   | ~£0.50–1  |
| Buffer                       | —        | ~£4–7     |
| **Total**                    |          | **£15–28**|

**Checkpoint rule**: push to HuggingFace Hub every 30 minutes during training.
A spot pod can be preempted at any time. Losing 6 hours of compute to a failed
checkpoint is a preventable mistake.

---

## Using nanochat

nanochat is the base codebase. We do **not** edit files in the `nanochat/` directory
directly. All modifications are in `mathnano/` and reference nanochat via import.

Key nanochat files to understand (Phase 0 reading list):
- `nanochat/model.py` — the transformer architecture
- `nanochat/tokenizer.py` — BPE tokenizer implementation
- `scripts/base_train.py` — the pretraining loop
- `scripts/sft_train.py` — the SFT training loop
- `scripts/grpo_train.py` — the GRPO RL loop
- `speedrun.sh` — the full pipeline in one script

The `--depth N` flag is the single most important knob. Start every session by
confirming which depth you're targeting.
