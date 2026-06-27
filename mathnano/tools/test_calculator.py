"""Tests for the calculator tool. `pytest mathnano/tools -q`."""
import pytest

from mathnano.tools.calculator import check_arithmetic, safe_arith


@pytest.mark.parametrize("expr,val", [
    ("48 + 24", 72), ("48 / 2", 24), ("5 * (3) + 3", 18), ("2 ** 5", 32),
    ("100 - 1", 99), ("(2 + 3) * 4", 20), ("7 % 3", 1), ("10 // 3", 3),
])
def test_safe_arith_ok(expr, val):
    assert safe_arith(expr) == val


@pytest.mark.parametrize("expr", ["x + 1", "import os", "foo()", "2 +", "", "__import__('os')"])
def test_safe_arith_rejects_nonarith(expr):
    assert safe_arith(expr) is None


def test_safe_arith_blocks_huge_power():
    assert safe_arith("2 ** 999999") is None     # would hang/oom -> refused


def test_verifies_correct_solution():
    sol = "48 / 2 = 24.\n48 + 24 = 72.\nThe answer is \\boxed{72}."
    r = check_arithmetic(sol)
    assert r["n_errors"] == 0 and r["corrected_answer"] is None
    assert r["n_checked"] >= 2


def test_flags_and_corrects_wrong_final_answer():
    sol = "48 / 2 = 24.\n48 + 24 = 73.\nThe final answer is \\boxed{73}."
    r = check_arithmetic(sol)
    assert r["n_errors"] == 1
    assert r["corrected_answer"] == "73" or r["corrected_answer"] == "72"
    # specifically: it should compute 72 and correct the boxed 73
    assert r["corrected_answer"] == "72"


def test_handles_latex_operators():
    sol = "Area = 6 \\times 7 = 42."
    r = check_arithmetic(sol)
    assert r["n_checked"] == 1 and r["checks"][0]["ok"]


def test_skips_symbolic_steps():
    # "5x = 15" has a variable -> not verifiable -> no false flag
    sol = "5x = 15\nx = 3"
    r = check_arithmetic(sol)
    assert r["n_errors"] == 0


def test_multiline_chained_equation_is_corrected():
    # the model splits the equation across lines and gets the product wrong (384*27 = 10368)
    sol = ("Total = widgets * days\n= 384 * 27\n= 10224.\nSo \\boxed{10224}")
    r = check_arithmetic(sol)
    assert r["n_errors"] == 1
    assert r["corrected_answer"] == "10368"


def test_no_equations():
    r = check_arithmetic("The answer is forty-two.")
    assert r["n_checked"] == 0 and r["corrected_answer"] is None
