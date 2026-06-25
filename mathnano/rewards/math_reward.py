"""Verifiable math reward — the single source of truth for "is this answer correct?".

Used by THREE consumers, so it must be robust and identical everywhere:
  1. Track B GRPO (TRL) — reward signal during RL.
  2. The evaluation harness (`mathnano/eval/`) — benchmark scoring.
  3. (Track A reuses nanochat's own `tasks/gsm8k.py` reward; we mirror its #### rule here so
     our eval of Track A is consistent with what it was trained against.)

WHY this matters: GRPO optimises *whatever this function rewards*. A loose extractor that
counts wrong answers as correct teaches the model to be confidently wrong; a strict one that
misses correctly-formatted answers starves the reward signal. So we (a) extract the model's
final answer with several fallbacks, (b) normalise away cosmetic LaTeX/formatting differences,
and (c) compare numerically / as fractions / symbolically before giving up on string equality.

The pipeline is: extract_answer(completion) -> normalise -> answers_equivalent(pred, gold).
Everything is pure-Python + optional sympy, so it is trivially unit-testable offline (see
serve/tests / mathnano/rewards tests) with no model or GPU.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

# sympy is optional: numeric/fraction/string paths work without it; it only adds symbolic
# equivalence (e.g. "1/2" == "0.5", "(x+1)^2" == "x^2+2x+1"). Guarded so importing this module
# never fails in a minimal env.
try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
    _SYMPY_TFM = standard_transformations + (implicit_multiplication_application,)
    _HAVE_SYMPY = True
except Exception:  # pragma: no cover - exercised only in minimal envs
    _HAVE_SYMPY = False


# ----------------------------------------------------------------------------------------
# 1. Answer extraction
# ----------------------------------------------------------------------------------------

_GSM8K_RE = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")
# "the answer is 42", "final answer: 42", "answer = \boxed{...}" handled separately.
_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=|are)?\s*",
    re.IGNORECASE,
)
# A trailing "= X" (last equation result) used as a weak fallback.
_TRAILING_EQ_RE = re.compile(r"=\s*([^=\n]+?)\s*\.?\s*$")
# Any number-like token (int/decimal, optional sign, optional thousands commas, optional %).
# The decimal part requires at least one digit so a sentence-ending "150." yields "150".
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def last_boxed(text: str) -> Optional[str]:
    r"""Return the content of the LAST ``\boxed{...}`` (or ``\fbox{...}``), brace-balanced.

    WHY brace-balanced and not regex ``\\boxed\{([^}]+)\}``: competition answers nest braces,
    e.g. ``\boxed{\frac{1}{2}}`` — a naive ``[^}]+`` stops at the first ``}`` and returns
    ``\frac{1`` . We scan for the matching close brace by depth.
    """
    for marker in (r"\boxed", r"\fbox"):
        idx = text.rfind(marker)
        if idx == -1:
            continue
        brace = text.find("{", idx)
        if brace == -1:
            continue
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace + 1 : i].strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    r"""Pull the model's final answer out of free-form text. Order = most → least reliable.

    1. ``\boxed{...}`` (competition / MATH convention).
    2. GSM8K ``#### X`` marker.
    3. An explicit "the answer is X" / "final answer: X" phrase (take its boxed/number/tail).
    4. A trailing ``= X``.
    5. The last number-like token anywhere (last resort).
    Returns None only if there is no number and no phrase at all.
    """
    if text is None:
        return None

    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()

    m = _GSM8K_RE.search(text)
    if m:
        return m.group(1).strip()

    # Look at the tail after the last answer-phrase, if present.
    phrase_iter = list(_ANSWER_PHRASE_RE.finditer(text))
    if phrase_iter:
        tail = text[phrase_iter[-1].end():].strip()
        # Prefer a boxed/number inside the tail; else take the first line/clause.
        inner_boxed = last_boxed(tail)
        if inner_boxed is not None:
            return inner_boxed.strip()
        nums = _NUMBER_RE.findall(tail)
        if nums:
            return nums[0].strip()
        if tail:
            return tail.splitlines()[0].strip().rstrip(".")

    m = _TRAILING_EQ_RE.search(text.strip())
    if m:
        return m.group(1).strip()

    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].strip()

    return None


# ----------------------------------------------------------------------------------------
# 2. Normalisation
# ----------------------------------------------------------------------------------------

_TEXT_WRAPPERS = re.compile(r"\\(?:text|mbox|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_STRIP_TOKENS = (
    r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\quad", r"\qquad",
    r"\$", "$", r"\%", "%", r"^\circ", r"\circ", r"\degree",
)
# Longer units first so "km/h" is consumed whole before "km" can match and strand "/h".
_UNIT_WORDS = re.compile(
    r"\b(?:km/h|m/s|dollars?|cents?|kilometers?|kilometres?|meters?|metres?|miles?|"
    r"hours?|minutes?|seconds?|days?|years?|degrees?|units?|percent|points?|apples?|"
    r"liters?|litres?|km|cm|mm|kg|g)\b",
    re.IGNORECASE,
)


def normalise(ans: str) -> str:
    r"""Canonicalise an answer string so cosmetic LaTeX/format differences don't fail a match.

    Removes math delimiters/spacing macros, unwraps ``\text{...}``, drops trailing units and
    punctuation, collapses ``\dfrac``/``\tfrac`` → ``\frac``, and strips thousands commas inside
    numbers. Does NOT lowercase (variable names are case-sensitive) and does NOT touch operator
    structure (that is the comparator's job).
    """
    if ans is None:
        return ""
    s = ans.strip()

    # Strip surrounding inline-math delimiters \( \) \[ \] and $...$.
    s = re.sub(r"^\\\(|\\\)$|^\\\[|\\\]$", "", s).strip()
    s = s.strip("$").strip()

    # Unwrap \text{...}/\mbox{...} etc. (repeat for nested simple cases).
    for _ in range(3):
        new = _TEXT_WRAPPERS.sub(r"\1", s)
        if new == s:
            break
        s = new

    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    for tok in _STRIP_TOKENS:
        s = s.replace(tok, "")

    s = _UNIT_WORDS.sub("", s)
    s = s.replace("\\", "\\")  # no-op keep; explicit
    # Thousands separators inside numbers: 1,000 -> 1000  (but keep tuples like (1,2)).
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ----------------------------------------------------------------------------------------
# 3. Comparison
# ----------------------------------------------------------------------------------------

def _to_number(s: str) -> Optional[float]:
    """Parse a scalar to float: plain number, percent, a/b, or \\frac{a}{b}. Else None."""
    if not s:
        return None
    t = s.strip()
    # \frac{a}{b}
    m = re.fullmatch(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", t)
    if m:
        try:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    # percent
    if t.endswith("%"):
        try:
            return float(t[:-1])
        except ValueError:
            return None
    # a/b
    if re.fullmatch(r"-?\d+\s*/\s*-?\d+", t):
        try:
            return float(Fraction(t.replace(" ", "")))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(t)
    except ValueError:
        return None


def _latex_to_sympy(s: str) -> str:
    """Best-effort LaTeX → sympy-parseable string for symbolic comparison."""
    t = s
    t = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", t)
    t = t.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    t = t.replace(r"\pi", "pi").replace("^", "**")
    t = re.sub(r"\\[a-zA-Z]+", "", t)  # drop any remaining macros
    t = t.replace("{", "(").replace("}", ")")
    return t.strip()


def _sympy_equal(a: str, b: str) -> bool:
    if not _HAVE_SYMPY:
        return False
    try:
        ea = parse_expr(_latex_to_sympy(a), transformations=_SYMPY_TFM, evaluate=True)
        eb = parse_expr(_latex_to_sympy(b), transformations=_SYMPY_TFM, evaluate=True)
        diff = sympy.simplify(ea - eb)
        return diff == 0
    except Exception:
        return False


def answers_equivalent(pred: Optional[str], gold: str, *, rel_tol: float = 1e-4,
                       abs_tol: float = 1e-6) -> bool:
    """True iff predicted answer matches ground truth, after normalisation.

    Tries, in order: exact normalised string, numeric (with tolerance), symbolic (sympy).
    Numeric tolerance exists because models write 0.3333 for 1/3; we don't want to punish that.
    """
    if pred is None:
        return False
    p, g = normalise(pred), normalise(gold)
    if p == g:
        return True

    np_, ng = _to_number(p), _to_number(g)
    if np_ is not None and ng is not None:
        return abs(np_ - ng) <= max(abs_tol, rel_tol * max(abs(np_), abs(ng)))

    return _sympy_equal(p, g)


# ----------------------------------------------------------------------------------------
# 4. Reward / correctness API
# ----------------------------------------------------------------------------------------

def is_correct(completion: str, ground_truth: str) -> bool:
    """Extract the answer from `completion` and check it against `ground_truth`."""
    return answers_equivalent(extract_answer(completion), ground_truth)


def math_reward(completion: str, ground_truth: str, *, correct: float = 1.0,
                incorrect: float = -1.0, no_answer: float = -1.0) -> float:
    """Binary verifiable reward for GRPO.

    `no_answer` (default = `incorrect`) lets you optionally penalise non-answers differently,
    e.g. set it to -1.5 to discourage the model from dodging. Defaults match nanochat's
    +1/-1 style so Track A and Track B share the same scale.
    """
    extracted = extract_answer(completion)
    if extracted is None:
        return no_answer
    return correct if answers_equivalent(extracted, ground_truth) else incorrect


if __name__ == "__main__":
    # Standalone sanity check (a few representative cases).
    checks = [
        (r"... so the total is \boxed{42}.", "42", 1.0),
        (r"The answer is **150 km**.", "150", 1.0),
        ("Step 1...\n#### 18", "18", 1.0),
        (r"Thus x = \boxed{\frac{1}{2}}.", "0.5", 1.0),
        (r"We get \boxed{7}.", "8", -1.0),
        ("I am not sure.", "5", -1.0),  # no number -> no_answer
    ]
    ok = 0
    for comp, gold, want in checks:
        got = math_reward(comp, gold)
        status = "OK" if got == want else "FAIL"
        ok += got == want
        print(f"[{status}] reward={got:+.0f} want={want:+.0f}  gold={gold!r}  <- {comp!r}")
    print(f"\n{ok}/{len(checks)} sanity checks passed "
          f"(sympy {'available' if _HAVE_SYMPY else 'NOT available'})")
