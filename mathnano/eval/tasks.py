"""Load benchmark test sets as `EvalRow`s. Answers verified to match the shared reward."""
from __future__ import annotations

from typing import Optional

from mathnano.eval.runner import EvalRow
from mathnano.rewards.math_reward import extract_answer

GSM8K_ID = "openai/gsm8k"
MATH_ID = "nlile/hendrycks-MATH-benchmark"


def load_gsm8k_test(limit: Optional[int] = None) -> list[EvalRow]:
    from datasets import load_dataset
    ds = load_dataset(GSM8K_ID, "main", split="test")
    rows = []
    for ex in ds:
        ans = extract_answer(ex["answer"])  # the `#### N` final answer
        if ans is not None:
            rows.append(EvalRow(problem=ex["question"], answer=ans))
        if limit and len(rows) >= limit:
            break
    return rows


def load_math_test(limit: Optional[int] = None) -> list[EvalRow]:
    from datasets import load_dataset
    ds = load_dataset(MATH_ID, split="test")
    rows = []
    for ex in ds:
        ans = (ex.get("answer") or "").strip()
        if ans:
            lvl = ex.get("level")
            rows.append(EvalRow(problem=ex["problem"], answer=ans,
                                level=int(lvl) if lvl is not None else None))
        if limit and len(rows) >= limit:
            break
    return rows


def load_task(name: str, limit: Optional[int] = None) -> list[EvalRow]:
    if name == "gsm8k":
        return load_gsm8k_test(limit)
    if name == "math":
        return load_math_test(limit)
    raise ValueError(f"unknown task {name!r} (use 'gsm8k' or 'math')")
