"""Unified eval CLI: score a model on GSM8K / MATH and write a report.

Examples:
    # real model (Track B):
    python -m mathnano.eval.run_eval --task all --backend hf --model Qwen/Qwen2.5-1.5B-Instruct
    # with a LoRA adapter from our SFT/GRPO:
    python -m mathnano.eval.run_eval --task gsm8k --backend hf --model Qwen/Qwen2.5-1.5B \
        --adapter track_b/outputs/grpo --limit 200
    # plumbing check, no ML deps:
    python -m mathnano.eval.run_eval --task gsm8k --backend dummy --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import time

from mathnano.eval.runner import DummyGenerator, evaluate
from mathnano.eval.tasks import load_task
from mathnano.seeds import set_seed


def build_generator(args):
    if args.backend == "dummy":
        return DummyGenerator()
    from mathnano.eval.backends import HFGenerator
    return HFGenerator(args.model, adapter=args.adapter, load_in_4bit=args.load_in_4bit)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a model on math benchmarks")
    ap.add_argument("--task", choices=["gsm8k", "math", "all"], default="all")
    ap.add_argument("--backend", choices=["hf", "dummy"], default="hf")
    ap.add_argument("--model", type=str, help="HF model id (hf backend)")
    ap.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap problems per task")
    ap.add_argument("--k", type=int, default=1, help="samples per problem (pass@k)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", type=str, default="experiments/eval_report.json")
    args = ap.parse_args()
    if args.backend == "hf" and not args.model:
        ap.error("--model is required with --backend hf")
    set_seed(1337)

    tasks = ["gsm8k", "math"] if args.task == "all" else [args.task]
    gen = build_generator(args)
    report = {"model": args.model, "adapter": args.adapter, "k": args.k,
              "limit": args.limit, "backend": args.backend, "tasks": {}}

    for name in tasks:
        print(f"\n=== {name} ===")
        rows = load_task(name, limit=args.limit)
        t0 = time.time()
        res = evaluate(rows, gen, k=args.k, max_new_tokens=args.max_new_tokens)
        print(res.summary())
        report["tasks"][name] = {
            "n": res.n, "correct": res.correct, "accuracy": res.accuracy,
            "by_level": {str(k): v for k, v in res.by_level.items()},
            "seconds": round(time.time() - t0, 1),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
