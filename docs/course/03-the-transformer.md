# 3. The transformer: embeddings, attention, and the residual stream

This is the heart of the machine. We'll build it in the order data flows: token ids → embeddings →
(many layers of attention + MLP) → a prediction. This chapter covers embeddings, the residual
stream, and **attention** (the key idea). The next chapter covers positions, normalization, and the
MLP.

Throughout, `B` = batch size, `T` = sequence length (number of tokens), `d` = `d_model` (the
model's "width"). ▶ In MathNano's depth-16 model, `d = 1024`.

## 3.1 Embeddings: ids → vectors

A token id is just an index. The **embedding table** is a learned matrix of shape `(V, d)`; looking
up row `i` gives a `d`-dimensional vector for token `i`.

```
token_ids:  (B, T)         e.g. [[464, 3797, 3332]]
embedding:  (B, T, d)      each id replaced by its learned d-vector
```

Why a learned vector and not the raw integer? Because "token 3797" has no meaning as a number —
3797 isn't "bigger" than 464 in any useful sense. The embedding lets the model place tokens in a
`d`-dimensional space where *geometry encodes meaning* (similar tokens end up nearby, directions
encode relationships). The model learns this space during training.

## 3.2 The residual stream: the model's "working memory"

Here's the mental model that makes transformers click. Think of the `(B, T, d)` tensor as a
**stream**: for each token position, a `d`-dimensional vector that flows straight through the whole
network. It starts as the token embedding. Every layer **reads** the stream, computes an update, and
**adds** it back:

```
x = embedding(tokens)
for each layer:
    x = x + attention(x)     # add a correction
    x = x + mlp(x)           # add another correction
prediction = unembed(norm(x))
```

Those `x = x + ...` lines are **residual connections**, and they are the most important lines in the
architecture. They mean each layer only has to learn a *correction* to the running representation,
not rebuild it from scratch. They also give gradients a clean highway straight back to the start
(Chapter 5), which is what makes networks dozens of layers deep trainable at all. The stream starts
as "raw token identity" and, layer by layer, accumulates context and abstraction until it's a rich
"what should come next here" representation.

## 3.3 Attention: letting tokens read each other

The MLP (next chapter) processes each position independently — it can't mix information between
tokens. **Attention is the only place tokens communicate.** It answers, for each position: "which
earlier positions are relevant to me, and what should I copy from them?"

### Query, Key, Value
From each token's stream vector, three vectors are computed by learned linear maps:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain / advertise?"
- **Value (V)** — "what information do I pass on if attended to?"

A position's new content is a **weighted average of all positions' Values**, where the weights come
from how well that position's Query matches each Key.

### Scaled dot-product attention, step by step
```
scores  = Q @ Kᵀ / sqrt(d_head)      # (T, T): how much position i should attend to position j
mask     applied so i can't see j>i  # causal mask (see below)
weights = softmax(scores, dim=-1)    # each row sums to 1 -> a distribution over positions
out     = weights @ V                # (T, d_head): each position's attended summary
```
- `Q @ Kᵀ` is every query dotted with every key → a `T×T` grid of raw relevance scores.
- **Why divide by `sqrt(d_head)`?** Without it, dot products grow with dimension, the softmax
  saturates (one weight ≈ 1, rest ≈ 0), and gradients vanish — attention "dies." Scaling keeps the
  scores' variance ~1 so the softmax stays soft and trainable.
- `softmax` turns scores into weights that sum to 1.
- `weights @ V` mixes the Values accordingly.

### The causal mask: no peeking at the future
We're training the model to predict the *next* token. If position 3 could attend to position 4, it
would be cheating — at generation time, token 4 doesn't exist yet. So we **mask** the upper triangle
of the score grid (set `scores[i, j] = -∞` for `j > i`) before the softmax, forcing each position to
attend only to itself and earlier positions. This is what makes it a **decoder-only / causal**
model — the design behind GPT, Llama, Qwen, and nanochat.

### Multi-head attention
One attention operation can only express one "kind" of relationship at a time. So we run several in
parallel — **heads** — each with its own smaller Q/K/V (`d_head = d / n_heads`). One head might
track subject–verb agreement, another might track "the number mentioned earlier in this word
problem." Their outputs are concatenated and projected back to width `d`.

```
Q,K,V:  (B, n_heads, T, d_head)
out:    (B, n_heads, T, d_head)  ->  concat heads ->  (B, T, d)  ->  output projection
```

▶ **In MathNano** depth-16 uses `n_heads = 8`, `d_head = 128`. We verified from nanochat's source
that it uses **full multi-head attention** (every head has its own K/V), *not* the
grouped-query-attention (GQA) our planning docs had assumed — a good reminder to read the code, not
the docs (Chapter 11). It also adds a couple of modern refinements (QK-normalization, a learned
attention scale) covered in [Chapter 4](04-rope-norm-mlp.md).

## 3.4 Cost: why context length is expensive
The score grid is `T×T`. Double the sequence length and attention does 4× the work and uses 4× the
memory. This `O(T²)` cost is *the* reason long-context is hard and why tricks like FlashAttention
(compute attention in tiles without ever storing the full grid) and sliding-window attention exist.

▶ **In MathNano** nanochat normally uses sliding-window attention, but our RTX 4090 (Ada
architecture) only had the slower SDPA attention path (FlashAttention-3 is Hopper-only), and SDPA
doesn't support the sliding window. We ran with full-context attention (`--window-pattern=L`), which
*doubled* our throughput (35% → 72% model-FLOPs-utilization). A real architecture-meets-hardware
decision we made live during the run.

## What breaks without this
- No **embeddings**: the model can't represent token meaning, only meaningless indices.
- No **residual stream**: deep networks become untrainable (gradients vanish) and each layer can't
  build on the last.
- No **attention**: tokens can't share information — the model can't relate "x" in the answer to "x"
  in the question, can't do anything that needs context. Attention is the transformer's defining
  organ.
- No **causal mask**: the model "sees the future" during training and collapses at generation time,
  when the future genuinely isn't there.

→ Next: [Positions, normalization, and the MLP](04-rope-norm-mlp.md)
