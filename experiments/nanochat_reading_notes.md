# nanochat Reading Notes (Phase 0)

Source: `karpathy/nanochat` @ master, cloned 2026-06-25 into `nanochat/`.
These are **verified-from-source** facts. Where they contradict our planning docs, the
source wins and the docs were corrected.

---

## 1. Architecture (`nanochat/gpt.py`) — what nanochat ACTUALLY is

nanochat is **not** the vanilla "LLaMA stack" our ARCHITECTURE.md teaches. It is a
modded-nanoGPT-style network. Header comment lists: rotary embeddings, QK norm, untied
embed/lm_head, **relu² MLP**, norm after embedding, **parameter-free RMSNorm**, no bias,
GQA support, Flash Attention 3.

| Component | Our docs said | nanochat actually does |
|---|---|---|
| Vocab size | 16,384 | **32,768 (2^15)**, padded to mult. of 64. Init loss ≈ ln(32768) ≈ **10.4** |
| Primary metric | cross-entropy loss | **bits-per-byte (bpb)** on val (vocab-invariant) |
| MLP activation | SwiGLU (3 matrices, 8/3 dim) | **relu²**: `F.relu(c_fc(x)).square()`, 4× expansion, 2 matrices |
| Norm | RMSNorm w/ learnable γ | **parameter-free** `F.rms_norm`, pre-norm, also after embedding |
| Embed / lm_head | weight-tied | **untied** (separate `lm_head` Linear) |
| Attention heads | GQA, n_kv_head=2 | depth-scaled config sets **n_kv_head = n_head (full MHA)**; GQA supported but unused by default |
| RoPE layout | interleaved even/odd (CLAUDE.md) / complex `freqs_cis` (ARCH.md) | **real-valued split-half**: `x1=x[:d], x2=x[d:]`, plus **QK-norm** and a 1.2 Q/K scale |
| Optimizer | Muon (2D) + AdamW (1D) | ✓ correct: `MuonAdamW`/`DistMuonAdamW` |

Extra tricks not in our docs: **value embeddings** (ResFormer, alternating layers),
**smear gate** (mix previous token embed), **backout** (subtract mid-layer residual),
per-layer **resid_lambdas / x0_lambdas**, **sliding-window attention** (`window_pattern`
default `"SSSL"`, final layer always full).

**Hardware note (critical for us):** Flash Attention 3 only runs on Hopper+; on the
**RTX 4090 (Ada)** nanochat uses the **SDPA fallback** in `nanochat/flash_attention.py`
(see `tests/test_attention_fallback.py`). `--fp8` requires H100+ → **not usable on 4090**.

## 2. The `--depth` parameter (`scripts/base_train.py:build_model_meta`)
```
model_dim = round_up(depth * aspect_ratio(=64), to head_dim(=128))
num_heads = model_dim // 128 ;  n_kv_head = num_heads  ;  n_layer = depth
```
| depth | n_embd | n_layer | n_head=n_kv_head |
|---|---|---|---|
| 8  | 512  | 8  | 4 |
| 12 (reference) | 768  | 12 | 6 |
| 16 (Track A main) | 1024 | 16 | 8 |
| 20 (nanochat default) | 1280 | 20 | 10 |

- `sequence_len` default **2048**.
- Training horizon: `--target-param-data-ratio` default **12** (Chinchilla=20). `num_iterations`
  auto-derived from data:param ratio, or set via `--num-iterations` / `--target-flops`.
- `--total-batch-size` auto (nearest power of 2), reference is d12.
- CLI flags use **dashes**: `--depth --aspect-ratio --head-dim --max-seq-len --window-pattern
  --num-iterations --target-flops --target-param-data-ratio --device-batch-size
  --total-batch-size --eval-every --core-metric-every --sample-every --save-every
  --resume-from-step --model-tag --run --fp8`. **No `--data_path`, no `--hf_repo`.**
- Documented tiny smoke test:
  `python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20`
- Resume: `--resume-from-step N`. wandb via `--run` (`dummy` disables).

## 3. Pretraining data (`nanochat/dataset.py`, `dataloader.py`, `tok_train.py`)
- Format: **parquet shards** `shard_NNNNN.parquet` with a single `text` column.
- Location: `$NANOCHAT_BASE_DIR/base_data_climbmix/` (`get_base_dir()` honors `NANOCHAT_BASE_DIR`).
  `list_parquet_files()` lists all `.parquet` there; `parquets_iter_batched()` uses all but the
  **last shard for train, last shard for val**.
- Default source: `karpathy/climbmix-400b-shuffle` (6543 shards) via `python -m nanochat.dataset -n N`.
- Tokenizer (`tok_train.py`) trains a **RustBPE** tokenizer from the same `text` iterator;
  default `--vocab-size 32768`, `--max-chars 2e9`, `--doc-cap 10000`. Saves to `base_dir/tokenizer`.
- Data is **tokenized on the fly** by the tokenizing dataloader. There is **no pre-tokenized
  uint16 `.bin`** anywhere — our DATASETS.md `.bin` pipeline was fiction.

**➜ MathPile integration (zero nanochat edits):** set `NANOCHAT_BASE_DIR=<ours>`, write MathPile
as `shard_00000.parquet …` (text column) into `$NANOCHAT_BASE_DIR/base_data_climbmix/` (reserve
last shard as val), **skip** the climbmix downloader, then `tok_train` + `base_train` as normal.

## 4. SFT (`scripts/chat_sft.py`) and RL (`scripts/chat_rl.py`)
- **SFT is task/Conversation-based**, not raw `{"messages": …}` JSONL. It mixes `tasks/` datasets
  (smoltalk, gsm8k, …) into `Conversation` objects. To add math SFT for Track A we add/extend
  `tasks/` classes, not write the JSONL format our docs describe.
- **RL is GRPO and HARDCODED to GSM8K**: `from tasks.gsm8k import GSM8K`; `train_task =
  GSM8K("main","train")`, `val_task = GSM8K("main","test")`. Reward = `train_task.reward(conv,
  text)`; correctness via `task.evaluate`. **Advantage = `rewards - rewards.mean()`** (mean
  baseline, **no std normalization**, no explicit KL term here). G = `device_batch_size` samples.
- `tasks/gsm8k.py` loads **`openai/gsm8k`**, extracts answer after `#### ` (strips commas),
  handles `<<…>>` calculator tool-call tags.

**➜ Track A RL plan:** reuse nanochat's built-in GSM8K GRPO directly. For MATH, add a
`tasks/math.py` Task subclass (mirror `gsm8k.py`) with `\boxed{}` extraction, and mix it in.

## 5. Checkpoints (`nanochat/checkpoint_manager.py`)
- `save_checkpoint` / `load_model`; saved under `base_dir` keyed by model tag (e.g. `d16`).
- **No built-in HuggingFace push** — we add a small uploader (push `base_dir` run folder to HF
  every ≤30 min) for spot-preemption safety.

## 6. Repo inventory worth knowing
`scripts/`: `tok_train tok_eval base_train base_eval chat_sft chat_rl chat_eval chat_cli chat_web`.
`tasks/`: `gsm8k mmlu arc humaneval smoltalk spellingbee common`. `nanochat/`: `gpt optim
tokenizer engine dataloader dataset checkpoint_manager core_eval loss_eval flash_attention fp8
report common ui.html`. There is a ready-made **web UI** (`scripts/chat_web.py` + `ui.html`) and a
**CLI chat** (`chat_cli.py`) — directly relevant to our Phase 6 product.

## Phase-0 open items — RESOLVED
- vocab=32768 ✓; MHA (n_kv_head=n_head) ✓; seq_len=2048 ✓; data dir constants + `NANOCHAT_BASE_DIR`
  ✓; checkpoint format ✓; HF push = none built-in ✓; RoPE = real split-half + QK-norm ✓.
- Still external (Phase 1): MATH mirror dataset id + schema; Track B base choice + VRAM.
