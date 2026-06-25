# MathNano — Runbook (turnkey execution order)

Everything offline-buildable is done and tested (74 passing tests). This runbook sequences the
remaining GPU/gated steps with concrete, verified commands. Tracks A and B are independent — you
can do B (the product) first if budget is tight.

Legend: 🖥️ local/CPU · 🟢 GPU pod · 🔑 needs your HF token · 💸 costs money

---

## 0. One-time
- 🖥️ `conda create -n mathnano python=3.11 && conda activate mathnano && pip install -r requirements.txt`
- 🔑 `huggingface-cli login` ; accept the MathPile license at huggingface.co/datasets/GAIR/MathPile.
- 🟢 On the RunPod RTX 4090 pod: `pip install -e nanochat/ && pip install -r requirements.txt`
  (Track B also: `pip install -r track_b/requirements.txt`). `git checkout` the pinned nanochat
  commit in `NANOCHAT_VERSION.txt`.
- 🟢 `export NANOCHAT_BASE_DIR=/workspace/nanochat_base`

## 1. Data (🖥️ CPU, mostly free)
```bash
# Inspect everything first (sanity):
python -m mathnano.data.inspect_data            # public sources
# GRPO + SFT sets (public; OMI subset streams):
python -m mathnano.data.prepare_grpo            # -> data/processed/grpo_problems.jsonl
python -m mathnano.data.prepare_sft             # -> data/processed/sft_combined.jsonl
# MathPile (🔑 gated, ~50GB) -> nanochat text shards:
huggingface-cli download GAIR/MathPile --repo-type dataset --local-dir data/raw/mathpile
python -m mathnano.data.prepare_mathpile --input data/raw/mathpile --target-tokens 4e9
python -m mathnano.data.inspect_data --mathpile-dir data/raw/mathpile   # verify SubSet/text
```

## 2. Track A — nanochat from scratch (🟢💸 the learning artifact)
```bash
# Tokenizer on MathPile text shards (vocab 32768):
python -m scripts.tok_train --vocab-size=32768
# Smoke (must see loss ~10.4 -> decreasing before paying for the real run):
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
  --depth=8 --device-batch-size=4 --num-iterations=20 --core-metric-every=-1
# Pretrain depth=16 (fallback 12 if budget). device-batch-size=4 ~17GB on a 4090:
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
  --depth=16 --device-batch-size=4 --save-every=1000 --model-tag=d16-mathpile
#   resume after preemption: add --resume-from-step=N
#   push checkpoints to HF every <=30 min (separate loop; nanochat has no built-in push).
# SFT then GRPO (GRPO is built-in on GSM8K; add tasks/math.py to also train on MATH):
python -m scripts.chat_sft   --model-tag=d16-mathpile ...
python -m scripts.chat_rl    --model-tag=d16-mathpile ...
```
> Track A SFT/RL feed data via nanochat `tasks/` classes, not our JSONL. TODO when on the pod:
> add `tasks/math.py` (mirror `tasks/gsm8k.py`, `\boxed` answer via our `math_reward`) and mix it
> into `chat_sft`/`chat_rl`.

## 3. Track B — Qwen2.5-1.5B SFT + GRPO (🟢💸 the shippable product)
Base confirmed: **Qwen/Qwen2.5-1.5B**. Stack: `track_b/requirements.txt` (TRL+PEFT+vLLM).
> `track_b/sft.py` and `track_b/grpo.py` are written **on the pod** so they can be smoke-tested
> against the exact pinned TRL/vLLM versions (this API is version-fragile — do not write blind).
> Inputs are ready: `sft_combined.jsonl` (chat, boxed answers) and `grpo_problems.jsonl`
> (`{problem,answer}`); reward = `mathnano.rewards.math_reward.math_reward`.
> Plan: LoRA SFT (~1–2h) → GRPO (G=4, vLLM rollouts, KL penalty) → merge/keep adapter.

## 4. Evaluate (🟢, cheap) — same harness for both tracks
```bash
python -m mathnano.eval.run_eval --backend dummy --task gsm8k --limit 10     # plumbing (🖥️)
python -m mathnano.eval.run_eval --task all --backend hf \
  --model Qwen/Qwen2.5-1.5B --adapter track_b/outputs/grpo --limit 500
```
Produces accuracy / per-level / pass@k and a JSON report. Run at each stage (base→SFT→GRPO,
pretrain→SFT→GRPO) for the headline comparison table; the key signal is **GRPO > SFT**.

## 5. Serve the product (🖥️ or 🟢)
```bash
MATHNANO_MODEL=Qwen/Qwen2.5-1.5B MATHNANO_ADAPTER=track_b/outputs/grpo \
  python -m uvicorn serve.api:app --port 8000        # http://localhost:8000
# or: docker build -f serve/Dockerfile -t mathnano . && docker run -e MATHNANO_MODEL=... -p 8000:8000 mathnano
```

## 6. Ship
HF model cards (note MathPile non-commercial license), push both models, README comparison
table + demo, failure analysis. See `PROJECT_PLAN.md` Phase 7.

---

### Budget guardrail
£25 is tight for both tracks on one 4090. **If it runs low, finish Track B + the product first;**
shorten Track A (depth=12 or stop pretrain early). Always spot instances; shut pods when idle;
checkpoint to HF ≤30 min.
