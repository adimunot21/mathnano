"""MathSolver — the product's inference layer.

Wraps a `Generator` (the same interface the eval harness uses, so the deployed model is the
exact thing we benchmarked) and adds product concerns: turn a problem into a structured result
``{solution, answer}`` by reusing the shared `extract_answer`, plus simple multi-turn chat and
streaming for the UI.

Backend-agnostic: tests inject a `DummyGenerator`; production builds an `HFGenerator` from env.
`build_default_solver()` never raises — if no model/ML stack is available it falls back to a dummy
so the server (and UI) always boot for development.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from mathnano.eval.runner import DEFAULT_SYSTEM, Generator
from mathnano.rewards.math_reward import extract_answer

# Chat-turn leak markers: small models sometimes generate past <|im_end|> into a new turn.
_LEAK_MARKERS = ("<|im_end|>", "<|im_start|>", "\nassistant", "\nuser", "\nsystem")


def _clean_solution(text: str) -> str:
    """Trim trailing rambling so the displayed solution ends at the answer.

    The SFT model occasionally runs past its boxed answer (it doesn't always emit the stop token).
    We cut at the first chat-turn leak marker, and—since our solutions end with the boxed answer—
    truncate at the end of the first line containing ``\\boxed{...}``.
    """
    cut = len(text)
    for marker in _LEAK_MARKERS:
        j = text.find(marker)
        if j != -1:
            cut = min(cut, j)
    text = text[:cut]
    i = text.find("\\boxed{")
    if i != -1:
        nl = text.find("\n", i)
        if nl != -1:
            text = text[:nl]
    # Strip trailing decode-leak junk (CJK / replacement chars) the model emits past its answer.
    text = re.sub(r"[一-鿿　-〿�\s]+$", "", text)
    return text.strip()


@dataclass
class Solution:
    answer: Optional[str]
    solution: str


class MathSolver:
    def __init__(self, generator: Generator, *, system: str = DEFAULT_SYSTEM,
                 max_new_tokens: int = 512, model_name: str = "unknown"):
        self.gen = generator
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

    def solve(self, problem: str, *, temperature: float = 0.0) -> Solution:
        text = self.gen.generate([problem], system=self.system, temperature=temperature,
                                 max_new_tokens=self.max_new_tokens, stop_on_boxed=True)[0]
        clean = _clean_solution(text)
        return Solution(answer=extract_answer(clean), solution=clean)

    def chat(self, messages: Sequence[dict], *, temperature: float = 0.0) -> Solution:
        """Single-response chat. We treat the last user turn as the problem.

        (A from-scratch 200M / 1.5B math model is not a general chatbot; the product is a math
        solver, so collapsing to the latest user question is the honest, robust behaviour.)
        """
        user_turns = [m["content"] for m in messages if m.get("role") == "user"]
        if not user_turns:
            return Solution(answer=None, solution="Ask me a math problem.")
        return self.solve(user_turns[-1], temperature=temperature)

    def stream(self, problem: str, *, temperature: float = 0.0,
               chunk: int = 24) -> Iterator[str]:
        """Yield the solution in chunks for a responsive UI.

        Real token-by-token streaming (HF `TextIteratorStreamer`) is a drop-in upgrade on the
        HF backend; this server-side chunking keeps the product shell backend-agnostic and lets
        the SSE endpoint + UI be fully tested with the dummy backend.
        """
        text = self.solve(problem, temperature=temperature).solution
        for i in range(0, len(text), chunk):
            yield text[i:i + chunk]


def build_default_solver() -> MathSolver:
    """Construct the production solver from env, falling back to a dummy if unavailable.

    Env: MATHNANO_MODEL (HF id), MATHNANO_ADAPTER (LoRA path), MATHNANO_4BIT=1.
    """
    model_id = os.environ.get("MATHNANO_MODEL")
    if not model_id:
        from mathnano.eval.runner import DummyGenerator
        print("[serve] MATHNANO_MODEL not set — using DummyGenerator (dev mode).")
        return MathSolver(DummyGenerator(), model_name="dummy")
    try:
        from mathnano.eval.backends import HFGenerator
        gen = HFGenerator(model_id, adapter=os.environ.get("MATHNANO_ADAPTER"),
                          device=os.environ.get("MATHNANO_DEVICE", "auto"),
                          load_in_4bit=os.environ.get("MATHNANO_4BIT") == "1")
        # Lower default cap keeps CPU latency reasonable (most solutions fit well under this).
        max_new = int(os.environ.get("MATHNANO_MAX_NEW_TOKENS", "384"))
        return MathSolver(gen, model_name=model_id, max_new_tokens=max_new)
    except Exception as e:  # noqa: BLE001
        from mathnano.eval.runner import DummyGenerator
        print(f"[serve] failed to load {model_id} ({e!r}); falling back to dummy.")
        return MathSolver(DummyGenerator(), model_name="dummy")
