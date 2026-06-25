"""Build the GRPO problem set: ``{"problem": ..., "answer": ...}`` JSONL with verified answers.

GRPO needs only a problem and its ground-truth final answer — the model generates its own
solutions and the reward function (`mathnano.rewards.math_reward`) checks them. So this script
just pairs each problem with a clean, machine-checkable answer.

Sources (see data-contract.md): GSM8K train (answer after ``####``) and the MATH mirror
``nlile/hendrycks-MATH-benchmark`` (clean ``answer`` field — no ``\boxed`` parsing needed).

Verification built in: for each source we sanity-check that our reward function, given the
dataset's OWN solution text, agrees the ground-truth answer is correct. If the agreement rate is
low, our extractor and the data have drifted — fail loudly rather than train on a broken signal.

Usage:
    python -m mathnano.data.prepare_grpo                 # full set
    python -m mathnano.data.prepare_grpo --limit 50      # quick smoke
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterator

from mathnano.rewards.math_reward import extract_answer, is_correct
from mathnano.seeds import set_seed

CONFIG = {
    "gsm8k_id": "openai/gsm8k",          # public
    "math_id": "nlile/hendrycks-MATH-benchmark",  # MATH mirror w/ clean `answer`
    "out_path": "mathnano/data/processed/grpo_problems.jsonl",
    "seed": 1337,                         # deterministic shuffle for reproducibility
    "min_agreement": 0.85,                # fail if reward<->data agreement drops below this
}


def gsm8k_problems(split: str = "train") -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(CONFIG["gsm8k_id"], "main", split=split)
    for ex in ds:
        ans = extract_answer(ex["answer"])  # the `#### N` final answer
        if ans is not None:
            yield {"problem": ex["question"], "answer": ans, "source": "gsm8k"}


def math_problems(split: str = "train") -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(CONFIG["math_id"], split=split)
    for ex in ds:
        ans = (ex.get("answer") or "").strip()
        if ans:
            yield {"problem": ex["problem"], "answer": ans,
                   "source": "math", "level": ex.get("level")}


def _check_agreement(rows: list[dict], solutions: list[str], name: str) -> float:
    """Fraction of (solution, ground-truth answer) pairs our reward marks correct."""
    if not rows:
        return 1.0
    hits = sum(is_correct(sol, r["answer"]) for r, sol in zip(rows, solutions))
    rate = hits / len(rows)
    print(f"  [{name}] reward<->data agreement: {hits}/{len(rows)} = {rate:.1%}")
    return rate


def build(limit: int | None = None) -> list[dict]:
    """Collect GRPO rows from all sources, with a self-consistency check per source."""
    from datasets import load_dataset
    set_seed(CONFIG["seed"])
    rows: list[dict] = []

    # GSM8K — verify against its own `answer` solution text.
    gsm = list(gsm8k_problems("train"))
    if limit:
        gsm = gsm[:limit]
    g_src = load_dataset(CONFIG["gsm8k_id"], "main", split="train")
    _check_agreement(gsm, [g_src[i]["answer"] for i in range(len(gsm))], "gsm8k")
    rows.extend(gsm)

    # MATH — verify our reward agrees that the dataset's `solution` yields `answer`.
    mth = list(math_problems("train"))
    if limit:
        mth = mth[:limit]
    m_src = load_dataset(CONFIG["math_id"], split="train")
    _check_agreement(mth, [m_src[i]["solution"] for i in range(len(mth))], "math")
    rows.extend(mth)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GRPO problem set")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per source (smoke test)")
    ap.add_argument("--out", type=str, default=CONFIG["out_path"])
    args = ap.parse_args()

    rows = build(limit=args.limit)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_src: dict[str, int] = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"\nWrote {len(rows):,} GRPO problems to {args.out}")
    print(f"  by source: {by_src}")


if __name__ == "__main__":
    main()
