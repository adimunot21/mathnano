# 11. The real workflow: cloud GPUs, budgets, reproducibility, and the bugs

Courses usually stop at the algorithms. But most of what determines whether a real project *succeeds*
is the workflow around them: not running out of money, not losing your work, reading source instead
of assuming, and debugging the unglamorous failures. This chapter is the stuff nobody tells you,
drawn directly from doing it.

## 11.1 Renting a GPU without losing your shirt

We trained on a single **RTX 4090 spot instance** on RunPod (~$0.70/hr). The whole project — two
training tracks, evals, serving — came to **~£13**, under the £25 budget. What kept it cheap:

- **Smoke-test before scale.** A depth-8, 20-step run (pennies) caught config/data bugs before the
  11.8-hour run (Chapter 6). The most important cost-control habit, full stop.
- **Measure throughput, then commit.** We watched the first ~30 steps' tokens/sec and ETA before
  letting the long run continue — and used MFU to find the 2× attention speedup (Chapters 3, 5).
- **Idle time is the silent budget killer.** A GPU bills whether or not it's computing. Shut it down
  between phases; chain stages so the GPU is never sitting idle waiting for you.
- **Right-size the work.** Don't train on 9.5B tokens when 2.5B suffices; don't do a 9-hour SFT when
  a focused 4-hour one (1 epoch, bigger batch) gets you there (Chapter 7).

## 11.2 Persistence: never lose a run

Spot instances can be **preempted** (killed) anytime, and SSH connections drop. We survived both
without losing work because:
- **Checkpoint to persistent storage** (a network volume that outlives the pod) every N steps — at
  worst you lose minutes.
- **Run long jobs in `tmux`** so they survive SSH disconnects (we got disconnected mid-run and the
  training kept going; we just reconnected and re-attached).
- **Back up the irreplaceable artifacts off-machine** — we pushed every model to HuggingFace, so even
  deleting the pod *and* its volume was safe. (And we *verified* the backups by listing the repo
  files — the first time we checked, the big weights file hadn't actually uploaded. Trust, but
  verify.)

## 11.3 Reproducibility

A result you can't reproduce isn't a result. Pinned dependency versions, fixed random seeds, configs
saved alongside outputs, and a pinned commit of the upstream `nanochat` code. ▶ We hit exactly the
pain this prevents: `vllm` pinned `torch==2.5.1` which pinned `sympy==1.13.1`, conflicting with our
`sympy==1.13.3` — a version-resolution failure that's trivial when pinned and maddening when not.
The version-sensitive RL stack (transformers + trl + vllm) is the kind of thing that *must* be a
pinned, tested set.

## 11.4 Read the source, not the docs

A recurring theme: our own planning docs (written from memory) were **wrong** about nanochat, and we
only found out by reading its code:
- The architecture: **relu²** MLP (not SwiGLU), **untied** embeddings, **MHA** (not GQA), vocab
  **32,768** (not 16,384), QK-norm, value embeddings — a "modded-nanoGPT," not the textbook stack.
- The data pipeline: **parquet text shards** tokenized on the fly (not a `uint16` binary file).
- The metric: **bits-per-byte** (so init loss is `ln(32768) ≈ 10.4`, not the 9.7 we'd assumed).
- Even external facts: the dataset is `GAIR/MathPile` with field `subset` (not `EleutherAI/mathpile`
  / `source`); the MATH benchmark had been taken down and needed a mirror; `huggingface-cli` was
  deprecated mid-project in favor of `hf`.

We wrote the verified facts into `experiments/nanochat_reading_notes.md` and corrected every doc.
**For any external dataset, API, or library: inspect the real thing — print the fields, read the
function signatures — before you write code against your assumptions.** It is the highest-ROI habit
in the whole project.

## 11.5 A field guide to the bugs we hit
A representative sample, because debugging *is* the job:
- **Output buffering**: piping training through `tee` block-buffered stdout, so the run looked
  "hung" for 5 minutes when it was fine. Fix: `PYTHONUNBUFFERED=1`. (First check whether it's
  *actually* stuck — `nvidia-smi` showed the GPU at 100%.)
- **`torchrun` ate our flag**: it prefix-matched `--run` to its own option. Fix: on a single GPU you
  don't need `torchrun` at all — plain `python -m ...` works and sidesteps it.
- **Network-filesystem installs**: building a venv on the network volume took 20 minutes / hung.
  Fix: put the venv on the pod's local disk; keep only data/checkpoints on the network volume.
- **OOM from the vocabulary**: SFT at batch 16 OOM'd because the loss over a 152k-token Qwen vocab
  is huge; halving the micro-batch (and accumulating) fixed it.
- **The stop-token saga** (Chapters 8, 10): the bug that kept coming back in different costumes.

None of these are in textbooks; all of them are normal. The skill isn't avoiding bugs — it's
diagnosing them from the actual error and applying the minimal fix.

## What breaks without this
Without the workflow discipline you blow the budget on idle GPUs, lose a 12-hour run to a dropped
connection, can't reproduce your own result next week, and waste days building against datasets/APIs
that don't behave the way you assumed. The algorithms get the glory; the workflow is what ships.

→ Next: [Reflection](12-reflection.md)
