r"""Build the SFT dataset as chat JSONL: ``{"messages": [system, user, assistant]}``.

This is the **Track B (Qwen)** format. Track A (nanochat) consumes data via its own ``tasks/``
classes, not this JSONL — but both are built from the same sources and the same standardised
solution format so the two models are comparable.

Design decision — every assistant solution ends with::

    The final answer is $\boxed{<ANSWER>}$.

WHY: the reward/eval extractor looks for ``\boxed{}`` first. If we train the model to always box
its final answer, (a) GRPO gets a clean reward signal, (b) eval doesn't lose correct answers to
formatting, and (c) the shipped product returns machine-checkable answers. GSM8K solutions (which
use ``#### N`` and ``<<calc>>`` tags) are rewritten into this format; MATH/OMI solutions get the
canonical line appended if they don't already end boxed with the right value.

Sources & mix (see data-contract.md): OpenMathInstruct-2 (bulk, streamed subset) + all GSM8K
train + all MATH train. Default mix ≈ 80/6/14 by count.

Usage:
    python -m mathnano.data.prepare_sft --limit 50     # smoke
    python -m mathnano.data.prepare_sft                # full (downloads OMI subset)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterator, Optional

from mathnano.rewards.math_reward import extract_answer, is_correct
from mathnano.seeds import set_seed

CONFIG = {
    "gsm8k_id": "openai/gsm8k",
    "math_id": "nlile/hendrycks-MATH-benchmark",
    "omi_id": "nvidia/OpenMathInstruct-2",
    "omi_n": 100_000,           # streamed subset of OpenMathInstruct-2
    "out_path": "mathnano/data/processed/sft_combined.jsonl",
    "seed": 1337,
    "system_prompt": (
        "You are a careful mathematician. Solve the problem step by step, "
        "then give the final answer in \\boxed{}."
    ),
}

_CALC_TAG = re.compile(r"<<[^>]*>>")          # GSM8K calculator annotations
_GSM_FINAL = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")
_FINAL_LINE = "The final answer is $\\boxed{{{}}}$."


def _with_boxed(solution: str, answer: str) -> str:
    """Ensure the solution ends with the canonical boxed-answer line for `answer`."""
    solution = solution.rstrip()
    canonical = _FINAL_LINE.format(answer)
    # If it already boxes the correct value at the end, leave it; else append the line.
    if is_correct(solution, answer) and "\\boxed" in solution.split("\n")[-1]:
        return solution
    return f"{solution}\n{canonical}"


def _gsm8k_solution(answer_field: str) -> tuple[str, Optional[str]]:
    """Turn a GSM8K `answer` (steps + <<tags>> + '#### N') into clean steps + final answer."""
    m = _GSM_FINAL.search(answer_field)
    final = m.group(1).replace(",", "").strip() if m else None
    steps = _GSM_FINAL.sub("", answer_field)
    steps = _CALC_TAG.sub("", steps).strip()
    return steps, final


def _msg(problem: str, solution: str) -> dict:
    return {"messages": [
        {"role": "system", "content": CONFIG["system_prompt"]},
        {"role": "user", "content": problem.strip()},
        {"role": "assistant", "content": solution.strip()},
    ]}


def gsm8k_examples(limit: Optional[int]) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(CONFIG["gsm8k_id"], "main", split="train")
    n = 0
    for ex in ds:
        steps, final = _gsm8k_solution(ex["answer"])
        if not final:
            continue
        yield _msg(ex["question"], _with_boxed(steps, final))
        n += 1
        if limit and n >= limit:
            return


def math_examples(limit: Optional[int]) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(CONFIG["math_id"], split="train")
    n = 0
    for ex in ds:
        ans = (ex.get("answer") or "").strip()
        if not ans:
            continue
        yield _msg(ex["problem"], _with_boxed(ex["solution"], ans))
        n += 1
        if limit and n >= limit:
            return


def omi_examples(limit: Optional[int]) -> Iterator[dict]:
    from datasets import load_dataset
    n_target = limit if limit else CONFIG["omi_n"]
    ds = load_dataset(CONFIG["omi_id"], split="train", streaming=True)
    n = 0
    for ex in ds:
        ans = (ex.get("expected_answer") or "").strip()
        if not ans:
            continue
        yield _msg(ex["problem"], _with_boxed(ex["generated_solution"], ans))
        n += 1
        if n >= n_target:
            return


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT chat JSONL")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples per source (smoke test); also caps OMI subset")
    ap.add_argument("--out", type=str, default=CONFIG["out_path"])
    ap.add_argument("--no-omi", action="store_true", help="skip OpenMathInstruct-2 (offline/fast)")
    args = ap.parse_args()
    set_seed(CONFIG["seed"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    counts = {"gsm8k": 0, "math": 0, "omi": 0}
    sample_ok = sample_tot = 0

    with open(args.out, "w") as f:
        sources = [("gsm8k", gsm8k_examples), ("math", math_examples)]
        if not args.no_omi:
            sources.append(("omi", omi_examples))
        for name, gen in sources:
            for rec in gen(args.limit):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[name] += 1
                # verify the assistant turn is self-checkable (extractable boxed answer present)
                if sample_tot < 300:
                    asst = rec["messages"][-1]["content"]
                    sample_ok += extract_answer(asst) is not None
                    sample_tot += 1

    total = sum(counts.values())
    print(f"\nWrote {total:,} SFT examples to {args.out}")
    print(f"  by source: {counts}")
    print(f"  extractable final answer in sample: {sample_ok}/{sample_tot} "
          f"({(sample_ok/max(sample_tot,1)):.1%})  (should be ~100%)")


if __name__ == "__main__":
    main()
