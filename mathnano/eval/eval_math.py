"""Thin CLI: evaluate a model on MATH test (per-level breakdown). See run_eval.py.

    python -m mathnano.eval.eval_math --model Qwen/Qwen2.5-1.5B-Instruct --limit 200
"""
from __future__ import annotations

import argparse

from mathnano.eval.backends import HFGenerator
from mathnano.eval.runner import evaluate
from mathnano.eval.tasks import load_math_test


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate on MATH test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=1)
    args = ap.parse_args()

    rows = load_math_test(limit=args.limit)
    gen = HFGenerator(args.model, adapter=args.adapter)
    print(evaluate(rows, gen, k=args.k).summary())


if __name__ == "__main__":
    main()
