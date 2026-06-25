# Track B — capable product model (Qwen2.5-1.5B: SFT → GRPO)

The shippable model. We SFT a general base (Qwen2.5-1.5B) on our math chat data, then GRPO it with
the verifiable reward so it learns to be *correct*, not just well-formatted. Output is a small LoRA
adapter the product (`serve/`) loads.

## Setup (separate venv — this stack pins torch 2.5.x, nanochat uses 2.9.x)

```bash
python -m venv /workspace/venv_b && source /workspace/venv_b/bin/activate
cd /workspace/mathnano
pip install -r track_b/requirements.txt
```

## Run order (from repo root, in the venv)

```bash
# 1. SFT — smoke first (5 steps) to validate the TRL/transformers versions, then full
python track_b/sft.py --max-steps 5
python track_b/sft.py                         # -> track_b/outputs/sft (LoRA adapter)

# 2. Evaluate the SFT model (baseline before RL)
python -m mathnano.eval.run_eval --task all --backend hf \
  --model Qwen/Qwen2.5-1.5B --adapter track_b/outputs/sft --limit 300

# 3. GRPO from the SFT adapter — smoke first, then full
python track_b/grpo.py --sft-adapter track_b/outputs/sft --max-steps 5
python track_b/grpo.py --sft-adapter track_b/outputs/sft   # -> track_b/outputs/grpo

# 4. Evaluate after GRPO — the headline is GRPO > SFT
python -m mathnano.eval.run_eval --task all --backend hf \
  --model Qwen/Qwen2.5-1.5B --adapter track_b/outputs/grpo --limit 300
```

## On-pod tuning notes (single 4090, 24 GB)

These scripts are written against `trl==0.16` but TRL's API drifts — **always run the 5-step
smoke first**. Likely knobs:
- **GRPO batch/divisibility**: `num_generations` must divide the effective batch. If TRL errors,
  adjust `--per-device-batch` / `--num-generations`.
- **Memory**: vLLM shares the GPU with training. If OOM, lower `--vllm-mem` (e.g. 0.25) or pass
  `--no-vllm` (slower HF generation). SFT alone (LoRA, 1.5B, bf16) fits easily.
- **assistant-only loss**: `sft.py` masks the prompt via `assistant_only_loss=True`. If the chat
  template lacks `{% generation %}` markers and TRL errors, pass `--no-assistant-only`.

The model is served by pointing the product at the adapter:
`MATHNANO_MODEL=Qwen/Qwen2.5-1.5B MATHNANO_ADAPTER=track_b/outputs/grpo` (see `serve/README.md`).
