# MathNano — RunPod Training Guide

Everything you need to run training on RunPod without wasting money.
The most expensive mistakes in cloud GPU training are preventable.

---

## ⚠️ Corrections (2026-06-25)

- **nanochat targets 8×H100.** On a single **RTX 4090 (Ada)** it works via gradient
  accumulation (reduce `--device-batch-size` to 4 → ~17 GB), using the **SDPA attention
  fallback** (FA3 is Hopper-only) and **no FP8**. depth=16 pretrain is realistically
  **~10–30 h**. Budget for two tracks is tight — see the priority rule in `PROJECT_PLAN.md`.
- **Setup**: `pip install -e nanochat/` then `pip install -r requirements.txt`. Track B's
  GRPO stack is separate: `pip install -r track_b/requirements.txt`.
- **Point nanochat at MathPile**: `export NANOCHAT_BASE_DIR=/workspace/nanochat_base` and put
  our parquet shards in `$NANOCHAT_BASE_DIR/base_data_climbmix/`. Do **not** run the climbmix
  downloader.
- **Real commands** (flags use dashes; the examples further down with `--data_path` / `--resume`
  / `--hf_repo` are obsolete):
  - Tokenizer: `python -m scripts.tok_train --vocab-size=32768`
  - Smoke: `torchrun --standalone --nproc_per_node=1 -m scripts.base_train --depth=8 --device-batch-size=4 --num-iterations=20 --core-metric-every=-1`
  - Pretrain: `torchrun --standalone --nproc_per_node=1 -m scripts.base_train --depth=16 --device-batch-size=4 --save-every=1000 --model-tag=d16-mathpile`
  - SFT / RL: `python -m scripts.chat_sft …` / `python -m scripts.chat_rl …`
  - Resume: `--resume-from-step N`. wandb via `--run` (`dummy` disables).
- **Checkpoints**: saved under `$NANOCHAT_BASE_DIR` by `checkpoint_manager`. There is **no
  built-in HF push** — run our uploader on a ≤30-min loop against the run folder.

---

## GPU Selection

**Use: RTX 4090 on Community Cloud**

| GPU             | VRAM  | Spot price | Notes                              |
|-----------------|-------|------------|------------------------------------|
| RTX 4090        | 24 GB | ~$0.34/hr  | **Our choice.** Fits depth=16.     |
| RTX 3090        | 24 GB | ~$0.22/hr  | Slower BF16, not recommended       |
| A100 40GB       | 40 GB | ~$1.19/hr  | 3.5× more expensive, not worth it  |
| A100 80GB       | 80 GB | ~$1.64/hr  | Only needed for depth=20+          |

**Always use spot instances (Community Cloud)**. Spot pods can be preempted
(stopped without warning), but they're 40–60% cheaper. With checkpointing
every 30 minutes, preemption loses at most 30 minutes of work.

Depth=16 with batch_size=8 and seq_len=1024 uses approximately 18–20 GB VRAM.
This fits in an RTX 4090 (24 GB) with 4–6 GB to spare.

---

## One-Time Setup

### Create RunPod account and deposit credits

1. Go to runpod.io, create account
2. Deposit: add £25 ($32). RunPod accepts credit/debit cards.
3. Under Account Settings, note your API key for CLI use

### Install RunPod CLI (optional but useful)

```bash
pip install runpod
runpod config  # Paste your API key
```

---

## Launching a Pod

### Via the web UI

1. Go to runpod.io/console/pods
2. Click "Deploy"
3. Select: Community Cloud, RTX 4090
4. Template: RunPod PyTorch (includes PyTorch + CUDA pre-installed)
5. Container disk: 75 GB (for data + checkpoints)
6. Ports: 22 (SSH), 8888 (Jupyter, optional)
7. Click "Deploy"

### After the pod launches

Click "Connect" → "SSH" to get the SSH command. It looks like:
```
ssh root@<pod-ip> -p <port>
```

**First thing to do on any new pod:**
```bash
# Verify GPU
nvidia-smi
# Should show: RTX 4090, 24576 MiB total, 0 MiB used

# Check disk space
df -h
# Should show 75 GB available

# Check Python + PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Should print: 2.x.x and True
```

---

## Setting Up the Environment

```bash
# Clone your repo
git clone https://github.com/your-username/mathnano.git
cd mathnano

# Clone nanochat
git clone https://github.com/karpathy/nanochat.git

# Install dependencies
pip install -e nanochat/   # Install nanochat as a package
pip install datasets huggingface_hub transformers

# Log in to HuggingFace (for pushing checkpoints)
huggingface-cli login
# Paste your HuggingFace access token

# Download processed data (if you uploaded it previously)
huggingface-cli download your-username/mathnano-data \
  --repo-type dataset \
  --local-dir mathnano/data/processed/
```

---

## Starting a Training Run

### Always use tmux

If your SSH connection drops, any process not in tmux dies. Training runs take
hours. Use tmux:

```bash
tmux new-session -s training

# Inside tmux, start training:
bash mathnano/config/pretrain_d16.sh 2>&1 | tee experiments/logs/pretrain_d16.log

# Detach from tmux: Ctrl+B then D
# Re-attach later: tmux attach -t training
```

### Before starting any run, verify:

```bash
# 1. GPU is free
nvidia-smi
# Used memory should be near 0

# 2. Data file exists and is correct size
ls -lh mathnano/data/processed/mathpile_train.bin
# Should be ~18 GB

# 3. First 100 token IDs look reasonable (not all zeros)
python -c "
import numpy as np
data = np.fromfile('mathnano/data/processed/mathpile_train.bin', dtype=np.uint16)
print(f'Total tokens: {len(data):,}')
print(f'First 20 token IDs: {data[:20]}')
print(f'Max token ID: {data.max()} (should be < vocab_size)')
"

# 4. Smoke test: run 10 steps to confirm loss decreases and no crash
bash mathnano/config/pretrain_d8.sh --max_steps=10
# Loss at step 1 should be ~9.7. Loss at step 10 should be lower.
```

---

## Monitoring a Run

### Loss curves

Loss should decrease. For MathPile pretraining:
- Step 0: loss ≈ 9.7 (ln(16384), random model)
- Step 100: loss ≈ 5–7
- Step 1000: loss ≈ 3–4
- Step 10000: loss ≈ 2.5–3.2

If loss:
- **Doesn't decrease after 200 steps**: data loading problem, learning rate too high,
  or model architecture bug. Stop the run and debug.
- **Goes NaN**: learning rate is too high, or numerical instability. Reduce LR by 10×.
- **Spikes dramatically then recovers**: occasional, acceptable. If it keeps spiking,
  your gradient clipping value may be too high.

### MFU (Model FLOP Utilisation)

nanochat logs MFU. For an RTX 4090 running depth=16:
- **Above 40%**: excellent.
- **30–40%**: good.
- **Below 20%**: something is wrong. Common causes:
  - Data loading bottleneck (CPU can't prepare batches fast enough)
  - Wrong batch size (too small = lots of GPU idle time)
  - Gradient accumulation too high

### Watch it live

```bash
# In a second tmux window:
watch -n 5 nvidia-smi              # GPU utilisation (aim for >95%)
tail -f experiments/logs/pretrain_d16.log  # Training log
```

---

## Checkpointing

### The rule: checkpoint every 30 minutes, push to HuggingFace

nanochat saves checkpoints automatically. Configure the checkpoint frequency:
```bash
# In your config file:
--save_interval=1000    # Save every 1000 steps
# At ~80 steps/min on RTX 4090, 1000 steps ≈ 12.5 min
```

### Push to HuggingFace after each checkpoint

Add this to your training script to push after each save:
```bash
# At the end of your training script (or in a separate tmux window):
while true; do
    sleep 1800  # Every 30 minutes
    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='experiments/runs/d16_pretrain/',
    repo_id='your-username/mathnano-pretrain-d16',
    commit_message='checkpoint'
)
print('Checkpoint pushed to HuggingFace')
"
done
```

### Or use the Hub push flag

nanochat supports pushing directly:
```bash
--hf_repo=your-username/mathnano-pretrain-d16
--hf_push_interval=2000  # Push every 2000 steps
```

---

## Resuming After Preemption

If your spot pod is terminated:

```bash
# Spin up a new RTX 4090 pod
# Re-run setup steps above

# Download latest checkpoint from HuggingFace
huggingface-cli download your-username/mathnano-pretrain-d16 \
  --local-dir experiments/runs/d16_pretrain/

# Resume training from latest checkpoint
bash mathnano/config/pretrain_d16.sh --resume=experiments/runs/d16_pretrain/latest
```

nanochat reads the latest checkpoint automatically if `--resume` points to
the checkpoint directory.

---

## Cost Management

### Before starting any run, estimate the cost

```python
# Quick calculator
hours_planned = 20  # depth=16 pretraining
rate_per_hour = 0.34  # RTX 4090 spot in USD
cost_usd = hours_planned * rate_per_hour
cost_gbp = cost_usd * 0.78  # approximate USD→GBP
print(f"Estimated cost: ${cost_usd:.2f} / £{cost_gbp:.2f}")
```

### Shut down the pod when not actively training

RunPod bills for idle time. If you're going to be away for more than 20 minutes:
```bash
# Ensure latest checkpoint is pushed to HuggingFace
# Then stop the pod from the RunPod web console (not just disconnect SSH)
```

### Don't leave a pod running overnight without monitoring

Set a reminder or use RunPod's budget alerts:
- Go to RunPod Settings → Billing Alerts
- Set an alert at 80% of your budget

### The emergency stop

If something goes wrong and you're burning money:
1. SSH in immediately: `Ctrl+C` to stop the training script
2. Or go to RunPod web console and click "Stop Pod"
3. Your checkpoint is safe — nanochat saves on graceful stop

---

## Data Transfer

### Uploading data to a pod

For processed data (tokenised MathPile), download from HuggingFace:
```bash
huggingface-cli download your-username/mathnano-data \
  --repo-type dataset \
  --local-dir /workspace/mathnano/data/processed/
```

This is faster than uploading from your local machine because RunPod pods have
excellent bandwidth to HuggingFace.

### If HuggingFace is slow (rare)

Use RunPod's network volume:
1. Create a Network Volume (persistent, available across pod restarts)
2. Mount it to `/workspace`
3. Upload data once, keep it across pod runs

This adds ~$0.07/GB/month but saves re-downloading time.

---

## Useful Commands Quick Reference

```bash
# Check GPU
nvidia-smi

# Check disk
df -h

# Start training in background
tmux new-session -s training "bash pretrain_d16.sh 2>&1 | tee train.log"

# Detach from tmux: Ctrl+B then D
# Re-attach: tmux attach -t training

# Watch training log
tail -f train.log

# Stop training gracefully (saves checkpoint)
# Find the PID: ps aux | grep python
kill -SIGINT <pid>

# Push checkpoint manually
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='experiments/runs/d16_pretrain/',
    repo_id='your-username/mathnano-pretrain-d16'
)
"

# Check token count in a binary data file
python -c "
import numpy as np
d = np.fromfile('mathpile_train.bin', dtype=np.uint16)
print(f'{len(d):,} tokens = {len(d)*2/1e9:.1f} GB')
"
```
