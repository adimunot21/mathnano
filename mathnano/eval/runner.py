"""Evaluation engine: score any model on math benchmarks via the shared verifiable reward.

The engine is deliberately decoupled from *how* text is generated: it takes a `Generator`
(anything with `.generate(prompts) -> list[str]`). That means the SAME harness scores Track B
(HF transformers), Track A (nanochat engine, wrapped), or the live product API — and it can be
unit-tested with a `DummyGenerator` and zero ML dependencies.

Metrics: greedy accuracy (n_samples=1) or pass@k (n_samples=k, sampling). Correctness is decided
by `mathnano.rewards.math_reward.is_correct` — the exact function GRPO optimises — so eval and
training never disagree about what "correct" means.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

from mathnano.rewards.math_reward import extract_answer, is_correct

DEFAULT_SYSTEM = (
    "You are a careful mathematician. Solve the problem step by step, "
    "then give the final answer in \\boxed{}."
)


class Generator(Protocol):
    """Anything that turns prompts into completions. `prompts` are user problem strings."""
    def generate(self, prompts: Sequence[str], *, system: str = DEFAULT_SYSTEM,
                 temperature: float = 0.0, max_new_tokens: int = 512) -> list[str]:
        ...


@dataclass
class EvalRow:
    problem: str
    answer: str
    level: Optional[int] = None


@dataclass
class EvalResult:
    n: int
    correct: int
    k: int
    by_level: dict = field(default_factory=dict)
    records: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def summary(self) -> str:
        metric = "accuracy" if self.k == 1 else f"pass@{self.k}"
        lines = [f"{metric}: {self.correct}/{self.n} = {self.accuracy:.1%}"]
        for lvl in sorted(self.by_level):
            c, t = self.by_level[lvl]
            lines.append(f"  level {lvl}: {c}/{t} = {c/t:.1%}" if t else f"  level {lvl}: -")
        return "\n".join(lines)


class DummyGenerator:
    """Test/offline generator. `oracle` decides each completion given the prompt.

    Default oracle echoes a fixed wrong answer. Pass a custom oracle (e.g. one that returns the
    gold solution) to drive the harness deterministically in tests.
    """
    def __init__(self, oracle: Optional[Callable[[str], str]] = None):
        self.oracle = oracle or (lambda p: "I think the answer is \\boxed{0}.")

    def generate(self, prompts, *, system=DEFAULT_SYSTEM, temperature=0.0, max_new_tokens=512,
                 **_):
        return [self.oracle(p) for p in prompts]


def evaluate(rows: Sequence[EvalRow], generator: Generator, *, k: int = 1,
             temperature: Optional[float] = None, max_new_tokens: int = 512,
             batch_size: int = 64, progress: bool = True) -> EvalResult:
    """Score `rows`. k=1 → greedy accuracy; k>1 → pass@k with sampling.

    pass@k = a row counts correct if ANY of its k samples is correct (standard convention).
    """
    temp = temperature if temperature is not None else (0.0 if k == 1 else 1.0)
    res = EvalResult(n=len(rows), correct=0, k=k)

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        # For pass@k we generate k completions per problem; flatten then regroup.
        prompts = [r.problem for r in batch for _ in range(k)]
        comps = generator.generate(prompts, temperature=temp, max_new_tokens=max_new_tokens)

        for i, row in enumerate(batch):
            group = comps[i * k:(i + 1) * k]
            hit = any(is_correct(c, row.answer) for c in group)
            res.correct += hit
            if row.level is not None:
                c, t = res.by_level.get(row.level, (0, 0))
                res.by_level[row.level] = (c + hit, t + 1)
            res.records.append({
                "problem": row.problem, "gold": row.answer, "level": row.level,
                "correct": hit, "predicted": extract_answer(group[0]),
                "completion": group[0],
            })
        if progress:
            done = min(start + batch_size, len(rows))
            print(f"  {done}/{len(rows)}  running acc={res.correct/done:.1%}", flush=True)

    return res
