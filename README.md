# MathNano

A mathematical-reasoning language model project built two ways:

- **Track A — from scratch.** A ~200M-parameter transformer ([nanochat](https://github.com/karpathy/nanochat))
  pretrained on math text (MathPile), then SFT, then GRPO with verifiable rewards. The goal here
  is *understanding every architectural and training decision from first principles.*
- **Track B — the product.** SFT + GRPO (LoRA, verifiable math reward) on a small pretrained base
  (Qwen2.5-1.5B) for a genuinely capable solver. This is the model the app ships.
- **Product.** A chat web UI + a clean inference API/CLI you can build on (`serve/`).

The headline result is the same mechanism behind DeepSeek-R1 at ~1/50th the scale: **GRPO with
verifiable binary rewards** (extract the final answer, check it against ground truth — no human
labels, no reward model) measurably improving correctness over the SFT baseline.

## Status

Phase 0 (grounding) complete: nanochat cloned and verified from source, planning docs corrected,
repo scaffolded. **Start here:** [`experiments/nanochat_reading_notes.md`](experiments/nanochat_reading_notes.md)
— the verified-from-source facts about how nanochat actually works.

## Documentation

| Doc | What it covers |
|---|---|
| `CLAUDE.md` | Project brief, architecture decisions, code standards (read first) |
| `PROJECT_PLAN.md` | Phases, two-track structure, budget |
| `ARCHITECTURE.md` | Transformer components from first principles (+ §13: how nanochat really differs) |
| `DATASETS.md` | Datasets, access, processing, formats |
| `RUNPOD.md` | Single-RTX-4090 training, checkpointing, cost control |
| `experiments/nanochat_reading_notes.md` | Ground-truth notes on the nanochat codebase |

## Layout

```
mathnano/   shared code: data prep, verifiable reward, eval harness, configs
track_b/    Qwen2.5 SFT + GRPO (TRL/PEFT/vLLM)
serve/      FastAPI inference API + web chat UI + Dockerfile
notebooks/  architecture deep-dive (RoPE, attention, training curves)
nanochat/   upstream base codebase (not edited; driven via NANOCHAT_BASE_DIR)
```

## Setup (local, Phases 0–2)

```bash
conda create -n mathnano python=3.11 -y && conda activate mathnano
pip install -r requirements.txt          # data + eval + serving
# nanochat + Track B GPU stacks are installed on the training pod (see RUNPOD.md)
```

Training requires a GPU (RunPod RTX 4090 spot). MathPile is gated — accept the license on its
[HuggingFace page](https://huggingface.co/datasets/GAIR/MathPile) and provide an HF token.

## License / data notes

Code: MIT. MathPile is **CC BY-NC-SA 4.0 (non-commercial)** — models trained on it inherit that
restriction; this is a portfolio/research project, documented in the model card.
