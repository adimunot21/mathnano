# 6. Pretraining: learning to speak math from raw text

**Pretraining** is stage one of the recipe: run the training loop (Chapter 5) over a huge pile of
raw text so the model learns the statistics of language and the patterns of the domain. The output
is a model that's a brilliant *autocompleter* — and nothing more. It can't follow instructions yet.
That's the next stage. Here we make it fluent.

## 6.1 The data: MathPile

To get a model that "speaks math," you pretrain on math. We used **MathPile** (`GAIR/MathPile`), a
~9.5-billion-token corpus of mathematical text: arXiv papers (the bulk), textbooks, ProofWiki,
math StackExchange, Wikipedia, and web math. The arXiv dominance is deliberate — research papers are
dense with notation, definitions, and multi-step proofs, exactly the reasoning structure we want.

▶ **In MathNano**, three real lessons came from the data:
- **Inspect before you integrate.** Our planning docs said the dataset was `EleutherAI/mathpile`
  with a field called `source`. Reality (after we actually downloaded and looked): it's
  `GAIR/MathPile`, gated under a non-commercial license, with the field spelled `subset` (lowercase).
  We only learned the *exact* schema by opening one file. Always look at the real bytes.
- **You don't need all the data.** 9.5B tokens is far more than a 200M model can use well (see
  scaling laws, Chapter 12). We sampled ~2.5B tokens.
- **Balance the mix.** Reading files in order would have made the budget ~85% arXiv. We wrote a
  sampler that round-robins across sources so textbooks/proofwiki/etc. are represented — a better
  diet than arXiv alone.

## 6.2 The data pipeline: text → shards → on-the-fly tokenization

nanochat reads pretraining data as **parquet shards** (files with a `text` column) and tokenizes
*on the fly* during training. So our job was: convert MathPile's `jsonl.gz` files into text parquet
shards, reserve the last as a validation split, and point nanochat at them (via the
`NANOCHAT_BASE_DIR` environment variable — zero edits to nanochat itself). Then train the tokenizer
(Chapter 2) on those shards, then pretrain.

(Our original plan assumed a "tokenize everything into one big `uint16` binary file" pipeline. The
real nanochat doesn't work that way at all — another docs-vs-source correction. Reading the source
saved us from building the wrong thing; see Chapter 11.)

## 6.3 The smoke test: never spend money blind

Before the real run, we ran a **depth-8, 20-step smoke test**. Purpose: confirm the loss starts at
`ln(V) ≈ 10.4` and *decreases*, the data loads, nothing OOMs, and the throughput is sane — all for
pennies. It passed (loss 10.40 → 10.09 in 20 steps, 2.7 GB memory). Only then did we launch the real
run. **This discipline — cheap end-to-end test before expensive scale — is the single most
money-saving habit in ML.**

## 6.4 The real run, and reading it

depth-16, ~2.5B tokens, on one RTX 4090. Key live decisions and observations:
- The `--window-pattern=L` attention fix (Chapters 3, 5) → 72% MFU, 66k tok/sec, ~11.8 h.
- Loss fell 10.4 → 1.44; **validation bits-per-byte bottomed at 0.731.**
- Checkpoints saved every 500 steps to a persistent volume — so a dropped SSH connection or a
  preempted pod cost nothing (we hit both; the run survived in `tmux`).
- We watched mid-training samples evolve from gibberish → grammatical → factually-correct
  completions ("gold → Au", "Friday → Saturday").

### The overfitting lesson
Our run reached `epoch 7` — it cycled the ~2.5B-token corpus seven times. The validation bpb hit
its minimum (0.731) partway through and then *rose* to 0.793 by the end: classic mild **overfitting**
from too many passes over too little unique data. Takeaway for next time: more unique tokens, or stop
at the bpb minimum. A real result, honestly recorded in `RESULTS.md`.

## 6.5 What pretraining gives you — and doesn't

After pretraining, our model writes coherent math, recalls facts, and continues proofs in plausible
style. Ask it "If 5x + 3 = 13, then x is ___" and it confidently writes "x is 5" — **wrong** (it's
2), but in the right *form*. That's the signature of a pretrained-only model: it has the language and
the patterns of math, but not the discipline to be *correct* or to *follow your instruction*. It
treats your question as text to continue, not a task to complete.

Fixing "follows instructions" is **SFT** (Chapter 7). Fixing "is actually correct" is **RL**
(Chapter 8). Pretraining built the foundation both stand on.

## What breaks without this
Without pretraining you have no foundation — SFT on a randomly-initialized model would need
astronomically more labeled data to teach it language *and* the task at once. Pretraining is where
the model cheaply absorbs the structure of the world (well, of text) from unlabeled data; everything
after is comparatively tiny, targeted polishing.

→ Next: [Supervised fine-tuning](07-sft.md)
