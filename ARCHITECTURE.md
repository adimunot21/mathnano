# MathNano — Architecture Reference

This is the course document. It explains every component of the transformer
architecture we are building, from first principles. It is meant to be read
alongside nanochat's model.py.

For each component:
- What it is
- The problem it solves (why we need it)
- The mathematical formulation
- The intuition
- The code pattern
- What we use instead of the original design, and why

---

## The Big Picture

A language model takes a sequence of tokens as input and outputs a probability
distribution over the next token. It does this autoregressively: generate one
token, append it to the input, predict the next, repeat.

The architecture we're building is a **decoder-only transformer**. "Decoder-only"
means there is no encoder — the model only processes its own output from left to
right. This is the design behind GPT, Llama, Mistral, and nanochat.

```
Input tokens: [1847, 293, 8, 4910, ...]
                ↓
Embedding table: map each token ID to a vector of size d_model
                ↓
RoPE: add positional information via rotation
                ↓
N × Transformer blocks:
    └─ Pre-RMSNorm → GQA (self-attention) → residual
    └─ Pre-RMSNorm → SwiGLU (FFN) → residual
                ↓
Final RMSNorm
                ↓
Language model head: linear(d_model → vocab_size) → logits
                ↓
softmax → probability distribution over next token
```

The residual connections (the "→ residual" steps) are critical. They add the
input of each block to its output. This creates a "residual stream" that passes
through all layers. Each layer reads from the stream and adds a correction. The
stream starts as token embeddings and becomes increasingly abstract predictions
as it passes through more layers.

---

## 1. Tokenisation and BPE

### The problem

Neural networks process numbers, not text. We need to convert strings to integer
sequences. The design choices are:
- **Character-level**: every character is one token. Long sequences, small vocab.
- **Word-level**: every word is one token. Short sequences, huge vocab. Fails on
  rare words.
- **Subword (BPE)**: somewhere in between. The standard choice.

### Byte Pair Encoding (BPE)

BPE starts with individual bytes (256 tokens) and iteratively merges the most
frequent adjacent pair into a new token. After N merges, you have a vocabulary
of 256 + N tokens.

Example with small vocabulary:

Corpus: "low low low lowest"
Initial tokens: [l, o, w, " ", l, o, w, e, s, t] (spaces as tokens)
Most frequent pair: (l, o) → merge to "lo"
Corpus: [lo, w, " ", lo, w, " ", lo, w, e, s, t]
Next most frequent pair: (lo, w) → merge to "low"
...and so on.

For mathematics, BPE creates tokens for common LaTeX commands:
- `\frac` likely becomes one token
- `\sqrt` likely becomes one token  
- `\leq`, `\geq`, `\in` likely become single tokens
- Common words like "therefore", "proof", "theorem" likely become single tokens

**Vocabulary size trade-off**: 
- Smaller vocab (8K): longer sequences, cheaper per token, worse on rare math notation
- Larger vocab (64K): shorter sequences, more parameters in embedding table, better coverage
- nanochat uses 16K for small models — a reasonable middle ground

**Weight tying**: the embedding table and the final linear projection (logits layer)
share the same weights. Intuition: if embedding["dog"] represents the concept of
"dog" as a vector, then the final layer should also map representations near that
vector back to high probability of predicting "dog". Tying the weights enforces
this symmetry and reduces parameter count.

---

## 2. Token Embeddings

Each token ID is looked up in a table of shape (vocab_size, d_model). This converts
the integer sequence into a continuous vector sequence.

```python
embedding_table = nn.Embedding(vocab_size, d_model)
# Input: token_ids of shape (batch, seq_len)
# Output: x of shape (batch, seq_len, d_model)
x = embedding_table(token_ids)
```

`d_model` is the "width" of the model — how many dimensions the model uses to
represent each token's meaning. For depth=16, d_model=1024.

At this stage, the vectors have no positional information. Token 5 "the" has
the same embedding regardless of whether it appears at position 0 or position 500.
Positional encoding fixes this.

---

## 3. Positional Encoding → RoPE

### The problem

The attention mechanism is position-agnostic: if you shuffle the input tokens,
the attention weights change but the mechanism has no built-in way to prefer
nearby tokens over distant ones. Positional encoding gives the model a way to
know where each token is.

### Original approach: sinusoidal embeddings

The 2017 transformer added a positional encoding vector to each token embedding:
```
x_pos = x + PE(position)
```
where PE is a fixed pattern of sines and cosines. This works but has no
generalisation to sequences longer than those seen in training.

### RoPE: Rotary Position Embedding

Instead of *adding* positional information to the token representation, RoPE
*rotates* the Query and Key vectors used in attention.

For a vector x at position m, apply a rotation matrix R(m·θ):

```
q_rotated = R(m · Θ) · q
k_rotated = R(n · Θ) · k
```

The rotation matrix for each pair of dimensions (2i, 2i+1) is:
```
R(m · θᵢ) = [[cos(m·θᵢ), -sin(m·θᵢ)],
              [sin(m·θᵢ),  cos(m·θᵢ)]]

θᵢ = 10000^(-2i / d_head)
```

**Why this is clever**: when you compute the attention score q · k, you get:
```
q_rotated · k_rotated = (R(m·Θ)·q) · (R(n·Θ)·k) = q · R((m-n)·Θ) · k
```

The result depends only on (m - n) — the *relative distance* between positions.
The model can learn to attend based on relative position without caring about
absolute position. This generalises to sequences longer than those seen in training.

**The frequency pattern**: 
- i=0: θ₀ = 1.0. This dimension rotates once per token. Sensitive to adjacent tokens.
- i=d/2-1: θ = 10000^(-1) = 0.0001. This dimension barely rotates. Sensitive to
  positions 10000 tokens apart.

The model gets a built-in multi-scale temporal representation — like a clock with
hands that tick at different speeds.

```python
def precompute_freqs_cis(d_head: int, max_seq_len: int, theta: float = 10000.0):
    # Compute rotation frequencies for each dimension pair
    # freqs: (d_head // 2,)  — one frequency per dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    
    # For each position (0 to max_seq_len), compute the angles
    # positions: (max_seq_len,)
    positions = torch.arange(max_seq_len)
    
    # Outer product: angle for each (position, dimension) pair
    # freqs_matrix: (max_seq_len, d_head // 2)
    freqs_matrix = torch.outer(positions, freqs)
    
    # Return as complex numbers: cos(angle) + i*sin(angle) = e^(i*angle)
    # WHY complex: applying a rotation in 2D is equivalent to complex multiplication
    freqs_cis = torch.polar(torch.ones_like(freqs_matrix), freqs_matrix)
    # freqs_cis: (max_seq_len, d_head // 2)
    return freqs_cis
```

---

## 4. Layer Normalisation → RMSNorm

### The problem

As activations pass through many layers, their magnitudes can grow or shrink
exponentially. A layer that amplifies by 1.1× would give 1.1^30 ≈ 17× after 30
layers. This makes learning unstable: the optimal learning rate changes constantly.

Normalisation pins the magnitude of activations at each layer, keeping the
distribution stable throughout training.

### LayerNorm (original transformer)

For a vector x of length d:
```
μ = mean(x)
σ = std(x)
LayerNorm(x) = γ ⊙ (x - μ) / σ + β
```
γ (scale) and β (bias) are learnable per-dimension parameters.

### RMSNorm (what we use)

For a vector x of length d:
```
rms = sqrt(mean(x²))
RMSNorm(x) = γ ⊙ x / rms
```
No mean subtraction. No learnable bias.

**Why skip the mean subtraction?** Empirically, it's unnecessary. The network
can learn to re-centre through the weights. Skipping it saves ~10% of the
normalisation compute and removes a source of numerical instability.

**Pre-norm vs post-norm**: we apply normalisation *before* attention and FFN
(pre-norm), not after (post-norm as in the original transformer). 

Pre-norm keeps the residual stream clean: the residual branch always adds to
an un-normalised stream. Post-norm normalises the sum, making the residual
contribution variable with layer depth. Pre-norm is more stable for deep models.

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # One learnable scale per dimension. No bias.
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps  # Prevents division by zero
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) — arbitrary leading dimensions
        # Compute RMS along the last dimension
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # rsqrt(y) = 1/sqrt(y) — fused operation, numerically stable
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for the norm computation, then back
        # WHY: BF16 arithmetic is lossy. The norm computation needs FP32 precision
        #      or you get different norm values depending on input scale.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

---

## 5. The Attention Mechanism

### Why attention?

A recurrent network (LSTM, GRU) processes tokens sequentially. To relate token
1 to token 100, the information must pass through 99 intermediate steps. This
makes long-range dependencies hard to learn.

Attention lets every token look at every other token in a single step. The
computational cost is O(seq_len²) but long-range dependencies become O(1).

### Scaled Dot-Product Attention

For each position in the sequence, compute three vectors from the token representation:
- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I contain?"
- **Value (V)**: "what information will I provide if attended to?"

The attention output at each position is a weighted average of all Values, where
the weights are the dot products of that position's Query with all Keys:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Q · Kᵀ produces a matrix of scores: score[i, j] = how much should position i
attend to position j? Dividing by √d_k prevents saturation. Softmax normalises
the scores to probabilities.

```python
def scaled_dot_product_attention(
    q: torch.Tensor,   # (batch, heads, seq_len, head_dim)
    k: torch.Tensor,   # (batch, heads, seq_len, head_dim)
    v: torch.Tensor,   # (batch, heads, seq_len, head_dim)
    is_causal: bool = True
) -> torch.Tensor:
    
    d_k = q.shape[-1]  # head_dim
    
    # Compute attention scores
    # (batch, heads, seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply causal mask: token at position i cannot attend to j > i
    # WHY: at inference time, future tokens don't exist. Training must match.
    if is_causal:
        seq_len = q.shape[-2]
        # Create upper triangular mask (True = mask out)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Normalise to probabilities
    # (batch, heads, seq_len, seq_len)
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    # (batch, heads, seq_len, head_dim)
    return torch.matmul(attn_weights, v)
```

### Multi-Head Attention (MHA)

Instead of one set of Q/K/V projections, use n_heads parallel attention operations,
each with smaller d_head = d_model / n_heads. Each head can specialise in different
types of patterns: syntactic, semantic, positional.

```python
# Project to Q, K, V
# d_model = n_heads * head_dim
q = self.wq(x).view(B, T, n_heads, head_dim).transpose(1, 2)
# q: (batch, n_heads, seq_len, head_dim)
k = self.wk(x).view(B, T, n_heads, head_dim).transpose(1, 2)
v = self.wv(x).view(B, T, n_heads, head_dim).transpose(1, 2)

# Attend (for each head independently)
output = scaled_dot_product_attention(q, k, v, is_causal=True)
# output: (batch, n_heads, seq_len, head_dim)

# Concatenate heads and project back
output = output.transpose(1, 2).contiguous().view(B, T, d_model)
# output: (batch, seq_len, d_model)
output = self.wo(output)
```

### Grouped Query Attention (GQA) — what we actually use

In MHA, each head has its own K and V projections. During autoregressive
generation, K and V for all previous positions must be cached. For 8 heads, that
means 8× memory for the KV cache.

GQA reduces this by assigning multiple Q heads to share a single K/V head pair.

With n_heads=8, n_kv_heads=2:
- Q heads: 8 (each with its own projection)
- K/V heads: 2 (4 Q heads share each K/V head)
- KV cache size: 4× smaller

```python
# Project Q to n_heads, K and V to n_kv_heads
q = self.wq(x).view(B, T, n_heads, head_dim).transpose(1, 2)
# q: (batch, 8, seq_len, head_dim)
k = self.wk(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
# k: (batch, 2, seq_len, head_dim)
v = self.wv(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
# v: (batch, 2, seq_len, head_dim)

# Expand K and V: repeat each K/V head (n_heads // n_kv_heads) times
groups = n_heads // n_kv_heads  # = 4
k = k.repeat_interleave(groups, dim=1)
# k: (batch, 8, seq_len, head_dim)  — now matches q shape
v = v.repeat_interleave(groups, dim=1)

# Attend normally
output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 6. The KV Cache

### The problem

At each generation step, the model computes attention over all previous tokens.
Without caching, generating a 1024-token response requires computing attention
over sequences of length 1, 2, 3, ..., 1024 — O(seq_len²) total work.

With the KV cache, we only compute K and V for the new token at each step. The
previous K and V tensors are stored in memory. Each step is O(seq_len) instead
of O(seq_len²) — linear in sequence length.

### What gets cached

The K and V tensors for each layer, for each previous position. Structure:

```python
# One cache entry per transformer layer
kv_cache = [
    (
        torch.zeros(1, n_kv_heads, max_seq_len, head_dim),  # keys
        torch.zeros(1, n_kv_heads, max_seq_len, head_dim),  # values
    )
    for _ in range(n_layers)
]
```

At each generation step, write the new K/V at position t, then read the full
cached K/V for attention.

### Why GQA matters for the cache

With n_kv_heads=2 instead of n_heads=8, the KV cache is 4× smaller. For a
depth=16 model with d_model=1024, seq_len=2048:

- MHA: n_layers × 2 × n_heads × seq_len × head_dim × 2 bytes
  = 16 × 2 × 8 × 2048 × 128 × 2 = 1.07 GB
- GQA: 16 × 2 × 2 × 2048 × 128 × 2 = 0.27 GB

At depth=16, this is manageable. At scale (70B params, 128 heads), the GQA
saving is the difference between fitting in memory or not.

---

## 7. Feed-Forward Network → SwiGLU

### The role of the FFN

After attention, which allows tokens to communicate, the FFN processes each
position independently. It is where the model does most of its "thinking" per
token — the heavy computation that transforms attended representations into
higher-level features.

The FFN is applied identically and independently to each position. Think of it
as a per-position MLP applied after the attention has mixed information.

### Standard FFN (GPT-2 style)

```python
FFN(x) = ReLU(x W₁) W₂
```
- W₁: d_model → 4*d_model  (expansion)
- W₂: 4*d_model → d_model  (projection back)
- ReLU: zero out negative activations

The 4× expansion is a heuristic that has worked empirically. The idea: expand
into a large space to have enough capacity, then project back.

### SwiGLU (what we use)

```python
FFN(x) = (Swish(x W₁) ⊙ (x W₂)) W₃
```
- W₁, W₂: d_model → hidden_dim  (two parallel projections)
- W₃: hidden_dim → d_model  (projection back)
- Swish: x * sigmoid(x)  (smooth, slightly negative near 0)
- ⊙: elementwise multiplication = the "gate"

The gate (x W₂) modulates the activated path (Swish(x W₁)). The network
learns, per dimension, how much of the activated signal to pass through. This
adaptive filtering consistently outperforms ungated activations at the same
parameter budget.

**Why Swish and not ReLU in the gate?** Swish is smooth and slightly negative for
negative inputs. Unlike ReLU, it doesn't hard-zero anything — it provides a
gradient everywhere. Combined with the gate, this gives more nuanced control
over the information flow.

**Dimension adjustment**: SwiGLU uses 3 matrices while standard FFN uses 2.
To keep parameter count equal, reduce hidden_dim to 8/3 of d_model:
- Standard FFN params: d_model × 4d_model + 4d_model × d_model = 8 d_model²
- SwiGLU params: 2 × (d_model × 8d/3) + (8d/3 × d_model) = 8 d_model²
Equal! (Rounded to nearest 64 for hardware efficiency.)

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # 8/3 of d_model, rounded to nearest multiple of 64
        # WHY round: modern GPUs process matrices in tiles of size 64 or 128.
        # Non-multiples waste hardware capacity.
        hidden = int(d_model * 8 / 3)
        hidden = (hidden + 63) // 64 * 64
        
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate (Swish)
        self.w2 = nn.Linear(d_model, hidden, bias=False)  # value
        self.w3 = nn.Linear(hidden, d_model, bias=False)  # project back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        gate = F.silu(self.w1(x))   # Swish(x W₁): (batch, seq_len, hidden)
        val  = self.w2(x)            # x W₂:        (batch, seq_len, hidden)
        return self.w3(gate * val)   # project back: (batch, seq_len, d_model)
```

---

## 8. The Transformer Block

One transformer block = pre-norm + attention + residual + pre-norm + FFN + residual.

```
x₀  = input to this block  (batch, seq_len, d_model)
       ↓
x₁  = x₀ + Attention(RMSNorm(x₀))    ← attention + residual
       ↓
x₂  = x₁ + FFN(RMSNorm(x₁))          ← FFN + residual
       ↓
output = x₂
```

The residual connections are the most important lines in this diagram. They are
what make very deep networks trainable. Without them, gradients would vanish over
many layers. With them, the gradient can flow backward directly through the residual
path without passing through attention or FFN.

The residual connections also create what researchers call the "residual stream".
Each block reads the current stream, computes an update, and adds it back. The
stream accumulates progressively more abstract representations of the token.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
    
    def forward(self, x: torch.Tensor, freqs_cis, kv_cache=None):
        # x: (batch, seq_len, d_model)
        
        # Attention sub-layer with pre-norm and residual
        # PRE-norm: normalise BEFORE passing to attention
        # RESIDUAL: add the original x back after attention
        x = x + self.attn(self.attn_norm(x), freqs_cis, kv_cache)
        # x: (batch, seq_len, d_model)  — same shape, updated values
        
        # FFN sub-layer with pre-norm and residual
        x = x + self.ffn(self.ffn_norm(x))
        # x: (batch, seq_len, d_model)
        
        return x
```

---

## 9. The Full Model

Stack N transformer blocks (N = `n_layers` from `--depth`), then apply a final
RMSNorm and project to logits:

```python
class MathNanoModel(nn.Module):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        
        # 1. Look up token embeddings
        x = self.embed(token_ids)
        # x: (batch, seq_len, d_model)
        
        # 2. Get rotary position encodings for these positions
        freqs_cis = self.freqs_cis[:seq_len]
        
        # 3. Pass through N transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis)
        # x: (batch, seq_len, d_model)
        
        # 4. Final normalisation
        x = self.norm_out(x)
        # x: (batch, seq_len, d_model)
        
        # 5. Project to vocabulary logits
        # Weight tying: reuse the embedding weights
        logits = F.linear(x, self.embed.weight)
        # logits: (batch, seq_len, vocab_size)
        
        return logits
```

---

## 10. Training Objectives

### Pretraining: Next-Token Prediction

Given a sequence [t₁, t₂, t₃, t₄], train the model to predict:
- t₂ given t₁
- t₃ given t₁, t₂
- t₄ given t₁, t₂, t₃

This is done efficiently: feed the full sequence through the model once (using
the causal mask), get logits for all positions, compute cross-entropy at each
position (except the first).

```python
def compute_loss(logits, targets):
    # logits:  (batch, seq_len, vocab_size)
    # targets: (batch, seq_len)  — the input shifted by one position
    
    # Flatten for cross-entropy
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=-1  # -1 = padding token, don't compute loss
    )
    return loss
```

At the start of training, the model has no knowledge. Loss = ln(vocab_size).
For vocab_size=16384: ln(16384) ≈ 9.7. This is your initial loss.

As training progresses, the model learns to predict the next token more
accurately. Loss gradually decreases toward ~2–3 for a good language model.

### SFT: Supervised Fine-Tuning

Same cross-entropy loss, but only on the assistant's tokens. The user turn is
provided as context (in the forward pass) but masked from the loss.

```python
# In the SFT dataloader, labels for user tokens are set to -1 (ignored)
# Only assistant token positions have valid label IDs
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), 
                       ignore_index=-1)
```

### GRPO: Reinforcement Learning

GRPO does not use cross-entropy. Instead:

1. For each problem, sample G completions: [c₁, c₂, ..., c_G]
2. Score each: r_i = reward(c_i, ground_truth)  → {+1, -1}
3. Normalise scores: ρᵢ = (rᵢ - mean(r)) / (std(r) + ε)
4. Compute loss: push up probability of high-ρ completions, push down low-ρ ones

```python
# GRPO loss (simplified)
# log_probs: (G,)  — log probability of each completion
# relative_rewards: (G,)  — ρ values

policy_loss = -(relative_rewards * log_probs).mean()
# This is the policy gradient: increase log_prob when reward is high
# A KL divergence penalty is added to prevent the policy from drifting too far
```

---

## 11. The Muon Optimizer

Standard AdamW maintains moving averages of the gradient (m) and the squared
gradient (v), then steps in the direction m / sqrt(v). For weight matrices, this
update has correlated components — similar rows get similar updates.

Muon orthogonalises the gradient update for 2D weight matrices. Orthogonalisation
makes the update directions independent, which speeds up convergence. It uses
the Newton-Schulz iteration, a numerically stable algorithm for matrix square root:

```
X_{k+1} = (3X_k - X_k³) / 2  (iterated 5 times)
```
This converges to a matrix with orthonormal rows, which is used as the update
direction for each weight matrix.

nanochat applies Muon to all 2D parameters (weight matrices) and AdamW to all
1D parameters (embedding table, RMSNorm learnable scales). You do not need to
implement Muon — nanochat ships with it — but you should know why it's there.

---

## 12. FlashAttention

Standard attention computes the full (seq_len × seq_len) attention matrix and
stores it in memory. For seq_len=2048 and n_heads=8, this is:
- 2048 × 2048 × 8 × 2 bytes (BF16) = 67 MB per batch element

FlashAttention rewrites the attention computation to never materialise this
full matrix. It processes the sequence in blocks that fit in GPU SRAM (fast
cache), computing attention incrementally. The result is mathematically identical
but:
- Memory: O(seq_len) instead of O(seq_len²)
- Speed: 2–4× faster due to better memory access patterns

nanochat uses PyTorch's built-in FlashAttention via:
```python
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
This automatically uses FlashAttention when available. You don't need to implement
it, but you should know it's happening and why it matters.

---

## Summary Table

| Component          | Old approach          | What we use      | Why                               |
|--------------------|-----------------------|------------------|-----------------------------------|
| Position encoding  | Learned absolute      | RoPE             | Generalises to longer sequences   |
| Normalisation      | LayerNorm (post-norm) | RMSNorm (pre-norm)| Faster, more stable gradients    |
| Activation         | ReLU / GeLU           | SwiGLU           | Better gradient flow, gating      |
| Attention heads    | Full MHA              | GQA              | 4× smaller KV cache               |
| Bias terms         | Linear layers w/ bias | bias=False       | Cleaner, no quality loss          |
| Optimiser          | AdamW                 | Muon + AdamW     | Faster convergence                |
| Attention compute  | Naive (O(n²) memory)  | FlashAttention   | 4× less memory, 2× faster         |

---

## 13. How nanochat ACTUALLY differs from this canonical stack

Everything above describes the *converged modern stack* (LLaMA/Mistral/Gemma/Qwen) and is
correct as pedagogy. But the code we train — Karpathy's nanochat (`nanochat/gpt.py`) — is a
**modded-nanoGPT** network and differs in concrete ways. Verified from source on 2026-06-25;
full detail in `experiments/nanochat_reading_notes.md`.

| This doc teaches | nanochat actually does | Why it matters |
|---|---|---|
| **SwiGLU** FFN (3 mats, 8/3 dim) | **relu²**: `F.relu(c_fc(x)).square()`, 4× expansion, 2 mats | Simpler, no gate; your SwiGLU notebook is "what others do", not nanochat |
| **Weight tying** embed↔lm_head | **Untied** (`lm_head` is its own Linear) | More params, slightly better; loss math unchanged |
| RMSNorm with learnable **γ** | **Parameter-free** `F.rms_norm`, also applied right after embedding | No γ to learn; one fewer thing to reason about |
| **GQA**, `n_kv_head=2` | **MHA**: depth-scaling sets `n_kv_head = n_head` | KV-cache is *not* shrunk in our config; GQA code path is dormant |
| RoPE: interleaved / complex `freqs_cis` | **Real split-half**: `x1=x[:d], x2=x[d:]` then rotate | Different memory layout; plus **QK-norm** + a 1.2 Q/K scale after RoPE |
| Vocab 16,384, init loss 9.7 | **Vocab 32,768**, init loss ≈ **10.4**, metric = **bits-per-byte** | Your "expected loss" sanity checks must use these numbers |

**Also present in nanochat but absent above** (worth understanding for the deep-dive):
- **Value embeddings (ResFormer)** on alternating layers, gated per head.
- **Smear gate** — mixes the previous token's embedding into the current one (cheap bigram info).
- **Backout** — subtracts a cached mid-layer residual before the final norm.
- Per-layer learnable scalars **`resid_lambdas`** (residual scale) and **`x0_lambdas`** (re-inject
  the initial embedding).
- **Sliding-window attention** (`window_pattern="SSSL"`; final layer always full context).
- **Flash Attention 3** on Hopper+; on our **RTX 4090 (Ada)** it uses the **SDPA fallback**
  (`nanochat/flash_attention.py`). `--fp8` requires H100+ and is unavailable to us.

**Recommendation for the Phase-2 notebooks:** implement the canonical components (RoPE, RMSNorm,
SwiGLU, GQA) for understanding *and* add a short notebook cell that reads `gpt.py` and points at
each real difference. The contrast (why nanochat chose relu²/untied/QK-norm/value-embeddings) is
itself a strong learning exercise.
