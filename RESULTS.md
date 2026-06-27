# MathNano — Results

Single RTX 4090 (RunPod, ~$0.70/hr), total compute spend ≈ **£13**. All artifacts on HuggingFace:
- Pretrain (Track A): [`adimunot/mathnano-pretrain-d16`](https://huggingface.co/adimunot/mathnano-pretrain-d16)
- SFT (Track B, **the product**): [`adimunot/mathnano-qwen1.5b-sft`](https://huggingface.co/adimunot/mathnano-qwen1.5b-sft)
- GRPO (Track B, failed run — archived): [`adimunot/mathnano-qwen1.5b-grpo`](https://huggingface.co/adimunot/mathnano-qwen1.5b-grpo)

## Track A — nanochat ~200M from scratch on MathPile
- depth=16, 32,768 vocab, 2.82 B tokens (~2.5 B unique, balanced across MathPile sources),
  ~11.8 h, **72% MFU** with `--window-pattern=L` (SDPA fallback on Ada; FA3 is Hopper-only).
- **Minimum validation bits-per-byte: 0.731** (mild overfitting in the final stretch — 7 epochs
  over the corpus; takeaway: use more unique tokens or stop at the bpb minimum next time).
- Generates coherent math/English text with real factual recall ("gold → Au", "Friday →
  Saturday") but does not reliably *solve* — expected for pretrain-only. Learning artifact, not
  the shipped product.

## Track B — Qwen2.5-1.5B: SFT → GRPO (eval on GSM8K + MATH, n=200 each)

| Stage | GSM8K | MATH |
|------:|:-----:|:----:|
| **SFT** | **39.0%** | **40.0%** |
| GRPO | 5.5% | 5.5% |

SFT MATH accuracy by level: L1 66.7% · L2 56.8% · L3 41.5% · L4 34.1% · L5 18.0%.

**SFT is the shipped model** — 39–40% is a strong result for a 1.5 B model fine-tuned on a £25
budget.

### Why GRPO collapsed the model (an honest, instructive finding)
GRPO optimises whatever the reward says is good, and our reward signal during GRPO was
**corrupted by the rollout setup**. On a single 4090 we couldn't run TRL's vLLM rollouts, so we
used the `--no-vllm` HF path — which generated **fixed-length 400-token completions that never
stopped at the answer**. Each rollout was "…\boxed{answer}… + ~250 tokens of rambling," and the
verifiable extractor (which reads the *last* number/box) frequently graded **correct solutions as
wrong**. GRPO then pushed the policy away from its good SFT behaviour → collapse (the flat/negative
training reward was the warning sign).

**Lesson:** GRPO is only as good as the reward's fidelity. The fix (future work) is clean,
stop-at-`<|im_end|>` rollouts (proper vLLM, or forcing eos in the HF generation path) and/or a
reward that reads the *first* boxed answer / truncates at the stop token. The eval harness already
stops correctly, which is why SFT measures accurately.

## Reproduce
See `RUNBOOK.md`. Serve the product: `MATHNANO_MODEL=Qwen/Qwen2.5-1.5B
MATHNANO_ADAPTER=adimunot/mathnano-qwen1.5b-sft python -m uvicorn serve.api:app` → http://localhost:8000.
