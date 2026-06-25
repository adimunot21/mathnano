# Data Contract

Verified-from-source field schemas for every dataset MathNano touches. "Inspect before
integrating" — these were confirmed by actually loading the datasets (2026-06-25), not from docs.
Re-run `python -m mathnano.data.inspect_data` to refresh.

---

## openai/gsm8k  (config `main`)  — SFT + GRPO + eval
- Source: `load_dataset("openai/gsm8k", "main")`. Public, no auth.
- Splits: **train 7,473 / test 1,319**.
- Fields: `question: str`, `answer: str`.
- `answer` contains worked steps with calculator tags `<<a*b=c>>` and the final answer after a
  `#### ` marker, e.g. `... #### 72`. Final answers are integers/simple decimals.
- **Answer extraction**: regex `#### (-?[0-9.,]+)`, then strip commas. (Same as nanochat's
  `tasks/gsm8k.py` and our `math_reward.extract_answer`.)

## nlile/hendrycks-MATH-benchmark  — MATH (primary mirror; original DMCA'd)  — SFT + GRPO + eval
- Source: `load_dataset("nlile/hendrycks-MATH-benchmark")`. Public, no auth.
- Splits: **train 12,000 / test 500**.
- Fields: `problem: str`, `solution: str` (LaTeX, contains `\boxed{}`), **`answer: str` (already
  extracted — use directly, no `\boxed` parsing)**, `subject: str`, `level: int` (1–5),
  `unique_id: str`.
- WHY this mirror: the clean `answer` field removes the nested-brace `\boxed` extraction risk for
  GRPO ground truth. Note test is a 500-problem subset (not the original 5,000) — fine at our scale.
- Fallback mirror: `qwedsacf/competition_math` (train 12,500 only; `level` as `"Level 5"` strings,
  `type` instead of `subject`, no extracted `answer`, `\boxed` lives in `solution`).

## nvidia/OpenMathInstruct-2  — SFT (bulk)
- Source: `load_dataset("nvidia/OpenMathInstruct-2", split="train")`. ~14M rows, large (~GBs) —
  **use `streaming=True`** and take a subset (we use ~100–200k).
- Fields: `problem: str`, `generated_solution: str` (CoT by Llama-3.1-405B), `expected_answer: str`,
  `problem_source: str` (e.g. `augmented_gsm8k`, `augmented_math`).

## GAIR/MathPile  — pretraining (Track A)  — GATED, NOT yet inspected here
- Source: gated, CC BY-NC-SA 4.0 (non-commercial). Accept license + `huggingface-cli login`,
  then **`huggingface-cli download GAIR/MathPile --repo-type dataset --local-dir <dir>`**
  (not `load_dataset`). Files are **jsonl.gz**.
- Expected fields (from dataset card; **verify on first download**): `text: str`,
  `SubSet: str` (one of arXiv, Textbooks, Wikipedia, ProofWiki, StackExchange, CommonCrawl),
  plus metadata (language scores, idx). ~9.5B tokens total.
- Our use: stream `text`, optionally filter/weight by `SubSet`, write **text parquet shards**
  (`shard_NNNNN.parquet`, single `text` column) into `$NANOCHAT_BASE_DIR/base_data_climbmix/`,
  last shard reserved as val. Subset to a Chinchilla-aware budget (depth=16 ⇒ ~2.4B tokens at the
  default 12 tokens/param, ~4B at ratio 20). nanochat tokenises on the fly; **no `.bin` step**.

---

## Standardised SFT solution format (our convention)
To make the reward extractor reliable after SFT/GRPO, every assistant solution we emit ends with:

```
The final answer is $\boxed{<ANSWER>}$.
```

WHY: GRPO and eval both call `math_reward`, which looks for `\boxed{}` first. Training the model
to always box its final answer means the reward signal is clean (not lost to format misses) and
the product's answers are machine-checkable. GSM8K solutions (which use `####`) are converted to
this format; MATH solutions already box, and we append the canonical line if missing.
