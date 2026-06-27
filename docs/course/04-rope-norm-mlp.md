# 4. Positions, normalization, and the MLP

Attention as described in Chapter 3 has a surprising hole: it's **order-blind**. Shuffle the input
tokens and the math barely changes — `softmax(QKᵀ)V` has no notion of "position 3 vs position 50."
But "dog bites man" ≠ "man bites dog." We need to inject position. We also need to keep activations
numerically stable (normalization) and to give the model per-token compute (the MLP). This chapter
finishes the transformer block.

## 4.1 Positional information → RoPE

### The problem and the old fix
The original (2017) transformer *added* a fixed pattern of sines/cosines to the embeddings so each
position had a signature. It worked but generalized poorly to lengths unseen in training.

### Rotary Position Embeddings (RoPE) — what modern models use
Instead of *adding* a position signal, RoPE **rotates** the Query and Key vectors by an angle
proportional to their position, before the dot product. Pair up the dimensions; for the pair at
"frequency" `θ_i`, a token at position `m` is rotated by angle `m·θ_i`.

The elegant consequence: when you take the dot product of a query at position `m` with a key at
position `n`, the rotations combine so the result depends only on the **relative** distance `m − n`,
not on `m` and `n` separately. The model learns "attend to the token 3 back" rather than "attend to
absolute position 42," which generalizes far better to new lengths.

The frequencies span scales: low-index dimension pairs rotate fast (sensitive to nearby tokens),
high-index pairs rotate slowly (sensitive to long-range structure) — a built-in multi-scale clock.

▶ **In MathNano** nanochat applies RoPE to Q and K inside every attention layer (real-valued,
split-half form), then adds two refinements we verified in its source: **QK-normalization**
(normalize Q and K before the dot product, which stabilizes attention) and a small constant scaling
of Q and K (sharper attention). These differ from the "textbook" stack — see `ARCHITECTURE.md §13`.

## 4.2 Normalization → RMSNorm

As activations pass through many layers, their magnitudes can drift (grow or shrink exponentially).
A layer that scales by 1.1× gives 1.1³⁰ ≈ 17× after 30 layers — training becomes unstable.
**Normalization** rescales activations at each sub-layer so the distribution stays steady and one
learning rate works throughout.

- **LayerNorm** (original): subtract the mean, divide by the standard deviation, then apply a learned
  scale and bias.
- **RMSNorm** (modern, what we use): skip the mean and the bias — just divide by the root-mean-square
  and (optionally) apply a learned scale: `x / sqrt(mean(x²) + ε)`. Empirically as good, ~10%
  cheaper, fewer things to go wrong.

**Pre-norm vs post-norm:** modern models normalize *before* each sub-layer (`x + attn(norm(x))`),
not after. Pre-norm keeps the residual highway (Chapter 3) clean and un-normalized, which is what
makes very deep stacks train stably.

▶ **In MathNano** nanochat uses **parameter-free RMSNorm** (not even a learned scale) and also
normalizes right after the embedding. Another verified-from-source delta from the generic stack.

## 4.3 The MLP (feed-forward network): per-token thinking

After attention mixes information *between* tokens, the **MLP** processes each token *independently*
— this is where most of the model's parameters live and most of its per-token "computation" happens.
The standard shape expands then contracts:

```
h = activation(x @ W1)   # d -> 4d   (expand into a wider space)
out = h @ W2             # 4d -> d   (project back)
```

The wide hidden layer (typically 4× `d`) gives the model room to compute nonlinear features; the
projection brings it back to stream width to be added to the residual.

**The activation** is the nonlinearity (without it, stacked linear layers collapse into one linear
layer and the model can't represent anything interesting). The lineage: ReLU → GeLU → **SwiGLU**
(a gated variant many LLaMA-family models use). ▶ **In MathNano**, nanochat actually uses **relu²**
(`relu(x)²`) — simpler than SwiGLU, two matrices instead of three. We'd assumed SwiGLU from the
generic stack; the source said otherwise. (The educational notebooks implement SwiGLU too, as "what
most models do," with a note pointing at the real code.)

## 4.4 The full transformer block

Putting it together, one block is:

```
x = x + attention(rmsnorm(x))   # communicate between tokens (+ RoPE, QK-norm inside)
x = x + mlp(rmsnorm(x))         # think per token
```

A model is just `N` of these stacked, then a final norm and the unembedding to vocabulary scores.
**Depth** = `N`. ▶ In MathNano, "depth-16" literally means 16 of these blocks; nanochat derives all
other dimensions (width, heads, learning rates, training length) from that single number using
scaling-law heuristics. We chose depth 16 (~200M params); depth 8 was our cheap smoke-test size.

The full forward pass, end to end:
```
ids (B,T) -> embed -> (B,T,d) -> [block]×N -> final norm -> unembed -> logits (B,T,V) -> softmax
```
Note the output is a distribution at **every** position at once (not just the last) — during training
that lets us learn from all `T` next-token predictions in one pass; during generation we only use
the last position's distribution.

## What breaks without this
- No **positional encoding**: the model is word-order-blind — "man bites dog" and "dog bites man"
  are identical to it. RoPE specifically buys generalization to longer sequences.
- No **normalization**: activations blow up or vanish across layers; deep models diverge or stall.
- No **MLP / nonlinearity**: the whole network collapses to a single linear map — no facts, no
  reasoning, just a glorified matrix multiply.

→ Next: [Training](05-training.md)
