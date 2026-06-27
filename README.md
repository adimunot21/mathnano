# MathNano

Two ways to build a mathematical-reasoning language model, end-to-end on a **single RTX 4090** for **~£13**:

- **Track A — from scratch.** A ~200M-parameter transformer ([nanochat](https://github.com/karpathy/nanochat)) pretrained from random init on math text (MathPile) — the deep-learning artifact, for understanding every architectural and training decision.
- **Track B — the product.** SFT (LoRA) on **Qwen2.5-1.5B** with a verifiable math reward → a genuinely capable step-by-step solver, served behind a clean API + web UI.

It also includes an **honest negative result**: a GRPO (RL) run that *collapsed* the model, with a full root-cause analysis — a real lesson in reward fidelity.

## Results

**Track B (the product), GSM8K + MATH, n=200 each:**

| Stage | GSM8K | MATH |
|------:|:-----:|:----:|
| **SFT (shipped)** | **39.0%** | **40.0%** |
| GRPO (failed run) | 5.5% | 5.5% |

SFT MATH by difficulty: L1 66.7% · L2 56.8% · L3 41.5% · L4 34.1% · L5 18.0%.

**Track A:** depth-16 nanochat, 32,768 vocab, ~2.5 B tokens, 72% MFU — **validation bits-per-byte 0.731**. Generates coherent math text with real factual recall; doesn't reliably *solve* (expected for pretrain-only).

Full write-up incl. the GRPO post-mortem: **[RESULTS.md](RESULTS.md)**.

## Live demo

```bash
conda create -n mathnano python=3.11 -y && conda activate mathnano
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft accelerate

MATHNANO_MODEL=Qwen/Qwen2.5-1.5B MATHNANO_ADAPTER=adimunot/mathnano-qwen1.5b-sft \
  MATHNANO_DEVICE=cpu python -m uvicorn serve.api:app --port 8000
```

Open **http://localhost:8000** — a chat UI with worked solutions and LaTeX rendering. (CPU answers take a few seconds; on a GPU it's instant.) Models load straight from HuggingFace.

## Models on HuggingFace
- [`adimunot/mathnano-qwen1.5b-sft`](https://huggingface.co/adimunot/mathnano-qwen1.5b-sft) — the product (LoRA adapter).
- [`adimunot/mathnano-pretrain-d16`](https://huggingface.co/adimunot/mathnano-pretrain-d16) — the from-scratch model.
- [`adimunot/mathnano-qwen1.5b-grpo`](https://huggingface.co/adimunot/mathnano-qwen1.5b-grpo) — the failed GRPO run (archived for the write-up).

## What's interesting here
- **One verifiable reward, used everywhere** (`mathnano/rewards/math_reward.py`, 64 tests): robust answer extraction (nested `\boxed{}`, `####`, units, fractions) + numeric/symbolic equivalence — the *same* function scores GRPO training and evaluation, so they can't disagree.
- **Model-agnostic eval + serving** via one `Generator` interface, so the deployed model is exactly the one benchmarked.
- **Verified-from-source integration**: nanochat's real architecture (relu² MLP, untied embeddings, QK-norm, MHA, vocab 32,768, parquet data) documented in `experiments/nanochat_reading_notes.md` after the planning docs got it wrong — see `ARCHITECTURE.md §13`.
- A reproducible, cost-disciplined cloud-GPU workflow in **[RUNBOOK.md](RUNBOOK.md)**.

## Layout
```
mathnano/   shared code: data prep, verifiable reward, eval harness
track_b/    Qwen2.5 SFT + GRPO (TRL/PEFT)
serve/      FastAPI inference API + web chat UI + Dockerfile
nanochat/   upstream base codebase (driven via NANOCHAT_BASE_DIR; not vendored)
```

## 📚 The course
A complete, first-principles course on how LLMs work — taught through this project, with every
concept tied to the real numbers and bugs from our run: **[docs/course/](docs/course/)**
(12 chapters, from "what is a language model" through the transformer, the pretrain→SFT→RL recipe,
evaluation, serving, and an honest reflection on scale).

## Docs
`CLAUDE.md` (project brief) · `PROJECT_PLAN.md` · `ARCHITECTURE.md` · `DATASETS.md` · `RUNPOD.md` · `RUNBOOK.md` · `RESULTS.md` · `docs/course/`.

## License / data
Code: MIT. The Track-A model is trained on **MathPile (CC BY-NC-SA 4.0, non-commercial)** and inherits that restriction. Track-B builds on Qwen2.5 (see its license). Research/portfolio project.
