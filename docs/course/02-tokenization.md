# 2. Tokenization: turning text into numbers

A neural network does arithmetic on vectors. Text is a string of characters. The **tokenizer** is
the bridge: it converts a string into a list of integers (and back). Get it wrong and nothing
downstream can work.

## 2.1 The three options

- **Character-level**: each character is a token. Tiny vocabulary (~hundreds), but sequences are
  very long (every letter is a step), so the model wastes capacity and compute on spelling.
- **Word-level**: each word is a token. Short sequences, but the vocabulary is huge and you choke
  on any word you didn't see in training (every typo, name, or `\theorem` is "unknown").
- **Subword (BPE)**: the middle ground everyone uses. Common words become single tokens; rare words
  break into pieces; *anything* is representable because the fallback is bytes. This is what we use.

## 2.2 Byte-Pair Encoding (BPE), concretely

BPE *learns* its vocabulary from data by repeatedly merging the most frequent adjacent pair:

1. Start with the raw bytes as the base vocabulary (256 tokens — so any text is encodable).
2. Count all adjacent token pairs in the corpus. Merge the most frequent pair into a new token.
3. Repeat until you reach the target vocabulary size.

Toy example on "low low low lowest":
```
start:  l o w _ l o w _ l o w _ l o w e s t
merge most frequent pair (l,o) -> "lo":   lo w _ lo w _ lo w _ lo w e s t
merge (lo,w) -> "low":                    low _ low _ low _ low e s t
```
After enough merges, frequent strings like "low", " the", "tion" — and, on math text, "\frac",
"\theorem", " proof" — become single tokens, while rare strings stay split.

▶ **In MathNano** we trained the tokenizer on our MathPile shards with nanochat's `tok_train.py`
(a fast Rust BPE) to a vocabulary of **32,768** (= 2¹⁵). Because it learned from *math* text, common
LaTeX and proof vocabulary compress to single tokens — which is the point of training a
domain-specific tokenizer rather than reusing a generic one.

## 2.3 Vocabulary size is a trade-off

- **Smaller vocab** (e.g. 8k): fewer parameters in the embedding/output tables, but text becomes
  *more tokens* (less compression) → longer sequences → more compute per document, and rare symbols
  fragment badly.
- **Larger vocab** (e.g. 128k): better compression (fewer tokens per document), but a bigger
  embedding table and a harder softmax over more classes.

32,768 is a reasonable middle for a small model. The frontier trend is larger vocabularies
(100k–256k) because compression matters more at scale.

## 2.4 Bits-per-byte: the metric that survives a vocab change

If you measure quality by average loss-per-token, you can't compare two models with different
vocabularies (a model with a bigger vocab has fewer, "easier" tokens). The fix is **bits-per-byte
(bpb)**: convert the loss into "how many bits does the model need to encode each *byte* of raw
text." Bytes are vocabulary-independent, so bpb is comparable across tokenizers. Lower = better.

▶ **In MathNano** bpb was our primary pretraining metric; the run reached a minimum validation
**bpb ≈ 0.731**. (For intuition: ~0.73 bits/byte means the model compresses math text to roughly a
tenth of its raw 8-bits-per-byte size — it has learned a lot of structure.)

## 2.5 Special tokens and chat formatting

Beyond text tokens, the vocabulary includes **special tokens** that mark structure — e.g.
`<|im_start|>`, `<|im_end|>` (Qwen's chat markers), an end-of-document token. These let us encode
*roles* in a conversation:
```
<|im_start|>system
You are a careful mathematician...<|im_end|>
<|im_start|>user
If 5x - 3 = 12, what is 5x + 3?<|im_end|>
<|im_start|>assistant
... \boxed{18}<|im_end|>
```
The model learns that text after `<|im_start|>assistant` is *its* turn, and that `<|im_end|>` ends
the turn. This "chat template" is how a raw text predictor becomes a turn-taking assistant
(Chapter 7).

▶ **In MathNano** these special tokens caused two of our most instructive bugs. During RL and CPU
serving, our model often *failed to emit* `<|im_end|>`, so generation never stopped — it rambled to
the length cap (≈400 tokens) every time. On the GPU that was just slow; in RL it **corrupted the
reward** (Chapter 8); on CPU it made "1+1" take two minutes until we added a rule to stop right
after the boxed answer (Chapter 10). Special tokens are small but they run the show.

## 2.6 The embedding and output tables are tied to the vocabulary

Two big parameter tables have a `V` dimension:
- the **embedding table**: shape `(V, d_model)` — maps each token id to a vector (Chapter 3);
- the **output/unembedding**: shape `(d_model, V)` — maps the final vector back to a score per token.

Some models *tie* these (share the weights); modern nanochat **does not** (they're separate). Either
way, this is why vocabulary size directly costs parameters.

## What breaks without this
With a bad tokenizer the model spends its capacity on spelling instead of meaning, sequences blow
up in length (cost), and rare-but-important symbols (every LaTeX command, every variable) fragment
into noise the model can't reason over. And without special tokens, there's no way to tell the model
"this is a question, now answer it" — it would just autocomplete the question.

→ Next: [The transformer](03-the-transformer.md)
