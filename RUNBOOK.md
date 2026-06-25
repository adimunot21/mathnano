# MathNano — Runbook (verified on a RunPod RTX 4090, 2026-06-26)

Turnkey execution order. Commands below are the ones that actually ran on the pod (corrected for
real-world gotchas: `hf` not `huggingface-cli`, plain `python` not `torchrun` on one GPU,
`--window-pattern=L` for the SDPA fallback). Tracks A and B are independent.

Legend: 🖥️ local/CPU · 🟢 GPU pod · 🔑 needs HF token · 💸 costs money

---

## 0. Pod setup (🟢)
```bash
# keep ALL heavy data on the persistent network volume
export HF_HOME=/workspace/hf_cache
export NANOCHAT_BASE_DIR=/workspace/nanochat_base
export HF_HUB_DISABLE_XET=1            # avoid an hf_xet shutdown segfault on CPU data jobs
mkdir -p "$HF_HOME" "$NANOCHAT_BASE_DIR"

cd /workspace
git clone https://github.com/adimunot21/mathnano.git && cd mathnano
git clone https://github.com/karpathy/nanochat.git
git -C nanochat checkout $(sed -n 2p NANOCHAT_VERSION.txt)

# nanochat is NOT pip-installable (-e fails on its flat layout) — install its deps directly and
# run scripts FROM the nanochat dir. This also upgrades torch to nanochat's pinned 2.9.1.
pip install "torch==2.9.1" "datasets>=4.0.0" "fastapi>=0.117.1" "kernels>=0.11.7" \
  "psutil>=7.1.0" "rustbpe>=0.1.0" "tiktoken>=0.11.0" "tokenizers>=0.22.0" \
  "uvicorn>=0.36.0" "wandb>=0.21.3"
pip install -r requirements.txt                     # our data/eval/serve deps
hf auth login                                       # 🔑 paste token; accept MathPile license online
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.9.1 / True
```

## 1. Data (🖥️ CPU on the pod; free)
```bash
# MathPile (🔑 gated). Big arXiv chunks; lands on the volume.
hf download GAIR/MathPile --repo-type dataset --local-dir /workspace/mathpile_raw
# SFT + GRPO sets (public; OMI subset streams). Run in a separate tmux window.
cd /workspace/mathnano
python -m mathnano.data.prepare_grpo      # -> data/processed/grpo_problems.jsonl  (~19.5k)
python -m mathnano.data.prepare_sft       # -> data/processed/sft_combined.jsonl   (~119k)
```

## 2. Track A — nanochat from scratch (🟢💸 the learning artifact)
```bash
# shard MathPile -> balanced text parquet shards (last = val), then train tokenizer
python -m mathnano.data.prepare_mathpile --input /workspace/mathpile_raw --target-tokens 2.5e9
cd /workspace/mathnano/nanochat
python -m scripts.tok_train --vocab-size=32768

# smoke (loss must start ~10.4 and drop): plain python on 1 GPU, NO torchrun
python -m scripts.base_train --depth=8 --device-batch-size=4 --total-batch-size=8192 \
  --num-iterations=20 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy

# real pretrain in tmux. --window-pattern=L is REQUIRED on the 4090 (SDPA has no sliding window;
# with L we got 72% MFU, 66k tok/sec, ~11.7h for depth=16 / ~2.82B tokens at $0.70/hr ≈ £6).
tmux new -s train
PYTHONUNBUFFERED=1 python -m scripts.base_train --depth=16 --window-pattern=L \
  --device-batch-size=4 --save-every=500 --core-metric-every=-1 \
  --model-tag=d16-mathpile --run=dummy 2>&1 | tee /workspace/d16_train.log
#   resume after preemption: add --resume-from-step=N   (checkpoints persist on the volume)

# then SFT + GRPO (GRPO is built-in on GSM8K; add tasks/math.py to also train on MATH)
python -m scripts.chat_sft --model-tag=d16-mathpile ...
python -m scripts.chat_rl  --model-tag=d16-mathpile ...
```

## 3. Track B — Qwen2.5-1.5B SFT + GRPO (🟢💸 the shippable product)
```bash
python -m venv /workspace/venv_b && source /workspace/venv_b/bin/activate
cd /workspace/mathnano && pip install -r track_b/requirements.txt
# smoke each step (5 steps) before the full run — see track_b/README.md
python track_b/sft.py  --max-steps 5  && python track_b/sft.py
python track_b/grpo.py --sft-adapter track_b/outputs/sft --max-steps 5 \
  && python track_b/grpo.py --sft-adapter track_b/outputs/sft
deactivate
```

## 4. Evaluate (🟢, cheap) — same harness for both tracks
```bash
python -m mathnano.eval.run_eval --backend dummy --task gsm8k --limit 10     # plumbing (🖥️)
python -m mathnano.eval.run_eval --task all --backend hf \
  --model Qwen/Qwen2.5-1.5B --adapter track_b/outputs/grpo --limit 500
```
Run at each stage (base→SFT→GRPO) for the comparison table; key signal is **GRPO > SFT**.

## 5. Serve the product (🖥️ or 🟢)
```bash
MATHNANO_MODEL=Qwen/Qwen2.5-1.5B MATHNANO_ADAPTER=track_b/outputs/grpo \
  python -m uvicorn serve.api:app --port 8000        # http://localhost:8000
# or: docker build -f serve/Dockerfile -t mathnano . && docker run -e MATHNANO_MODEL=... -p 8000:8000 mathnano
```

## 6. Ship
HF model cards (note MathPile non-commercial license), push both models, README comparison table +
demo, failure analysis. See `PROJECT_PLAN.md` Phase 7.

---

### Cost discipline ($0.70/hr observed)
Projected total ~£12–15 for both tracks. The lever is **idle time** — shut the pod the moment a
phase ends, and chain Track B right after Track A. Checkpoints live on the 150 GB network volume,
so a preempted/stopped pod loses nothing; remount the volume and resume.
```
