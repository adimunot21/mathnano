r"""A calculator tool for inference-time arithmetic verification & correction.

The small SFT model reasons correctly but slips on arithmetic ("48 + 24 = 73"). This tool reads the
model's own solution, recomputes every arithmetic equation it wrote with an exact, **safe** engine,
flags/corrects mistakes, and — when the final boxed answer is itself a direct computation that the
model got wrong — corrects the final answer too.

"Tool use" here = a calculator the serving layer calls on the model's output. It is *not* code
execution: we parse each expression to a Python AST and evaluate only numeric arithmetic nodes
(`+ - * / // % **`, unary +/-, parentheses). No names, calls, attributes, imports, or builtins are
ever evaluated, so there is nothing unsafe to run.

Public API:
    check_arithmetic(solution) -> {"checks", "n_checked", "n_errors", "corrected_answer"}
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Optional

from mathnano.rewards.math_reward import extract_answer

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_MAX_POW = 1e6  # refuse absurd exponents so `2**99999` can't hang/oom


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > _MAX_POW):
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def safe_arith(expr: str) -> Optional[float]:
    """Evaluate a pure-arithmetic expression, or return None if it isn't one."""
    expr = expr.strip()
    if not expr:
        return None
    try:
        return _eval(ast.parse(expr, mode="eval"))
    except Exception:
        return None


# LaTeX / unicode math -> python-arithmetic
def _normalize(text: str) -> str:
    t = text
    # Merge chained equations split across lines ("...= 384 * 27\n= 10224") into one line, so the
    # whole `a = b = c` chain is checkable. Also handle LaTeX align separators (& and \\).
    t = t.replace("&=", "=").replace(r"\\", "\n")
    t = re.sub(r"\n\s*=", " = ", t)
    t = t.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    t = t.replace("×", "*").replace("÷", "/").replace("·", "*")
    t = t.replace(r"\left", "").replace(r"\right", "").replace(r"\,", "")
    t = t.replace("^", "**").replace("$", "")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)           # drop remaining LaTeX macros
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)     # 1,000 -> 1000
    # implicit multiplication: 5(3) -> 5*(3), )( -> )*(, )5 -> )*5
    t = re.sub(r"(\d)\s*\(", r"\1*(", t)
    t = re.sub(r"\)\s*\(", ")*(", t)
    t = re.sub(r"\)\s*(\d)", r")*\1", t)
    return t


_HAS_OP = re.compile(r"[+\-*/%]")
_ARITH_TAIL = re.compile(r"[0-9.\s()+\-*/%]+$")


def _fmt(v: float) -> str:
    return str(int(v)) if abs(v - round(v)) < 1e-9 else f"{v:.6g}"


def _arith_candidate(seg: str) -> Optional[str]:
    """Extract the trailing pure-arithmetic expression from a segment (strips leading prose).

    "We get 48 / 2" -> "48 / 2". Rejects continuations of a symbolic expression: "5x + 3" yields
    a tail of "+ 3" which starts with a binary operator, so we drop it (we can't verify an
    expression that contains a variable).
    """
    m = _ARITH_TAIL.search(seg)
    if not m:
        return None
    cand = m.group(0).strip()
    if not cand or not re.search(r"\d", cand):
        return None
    if cand[0] in "+*/%":          # tail attached to a preceding variable/expr -> not standalone
        return None
    return cand


def check_arithmetic(solution: str) -> dict:
    """Verify the arithmetic in `solution`; return checks + an optionally-corrected final answer.

    A "check" is a pair of `=`-separated segments on one line where at least one side is a real
    arithmetic expression and both sides evaluate. If they disagree, it's flagged. If the model's
    final answer equals a flagged-wrong *stated* value, we return the computed value as the fix.
    """
    checks: list[dict] = []
    for line in _normalize(solution).split("\n"):
        if "=" not in line:
            continue
        segs = line.split("=")
        cands = [_arith_candidate(s) for s in segs]
        vals = [safe_arith(c) if c else None for c in cands]
        for i in range(len(segs) - 1):
            a, b = vals[i], vals[i + 1]
            if a is None or b is None:
                continue
            ai, bi = _HAS_OP.search(cands[i]), _HAS_OP.search(cands[i + 1])
            if not (ai or bi):
                continue  # "72 = 72" with no computation — nothing to verify
            # the side with an operator is the authoritative computation
            if ai:
                computed, stated, expr = a, b, cands[i]
            else:
                computed, stated, expr = b, a, cands[i + 1]
            checks.append({
                "expr": expr.strip(), "computed": _fmt(computed), "stated": _fmt(stated),
                "ok": abs(computed - stated) < 1e-6,
            })

    n_errors = sum(not c["ok"] for c in checks)

    # Correct the final answer iff it exactly matches a flagged-wrong stated value.
    corrected_answer = None
    ans = extract_answer(solution)
    a_num = safe_arith(ans) if ans is not None else None
    if a_num is not None:
        for c in checks:
            if not c["ok"] and abs(float(c["stated"]) - a_num) < 1e-6:
                corrected_answer = c["computed"]
                break

    return {"checks": checks, "n_checked": len(checks), "n_errors": n_errors,
            "corrected_answer": corrected_answer}


if __name__ == "__main__":
    demo = "We get 48 / 2 = 24.\nThen 48 + 24 = 73.\nThe final answer is \\boxed{73}."
    r = check_arithmetic(demo)
    print(f"checked={r['n_checked']} errors={r['n_errors']} corrected={r['corrected_answer']}")
    for c in r["checks"]:
        print(("  OK  " if c["ok"] else " FIX ") + f"{c['expr']} = {c['stated']} (calc {c['computed']})")
