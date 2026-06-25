"""Tests for the verifiable math reward. Run: `pytest mathnano/rewards -q`.

These guard the correctness core that GRPO optimises against. Cases are grouped by concern:
extraction, normalisation, equivalence, and the end-to-end reward. >50 cases total.
"""
import pytest

from mathnano.rewards.math_reward import (
    answers_equivalent,
    extract_answer,
    is_correct,
    last_boxed,
    math_reward,
    normalise,
)

try:
    import sympy  # noqa: F401
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    (r"so the total is \boxed{42}.", "42"),
    (r"\boxed{-7}", "-7"),
    (r"answer: \boxed{\frac{1}{2}}", r"\frac{1}{2}"),          # nested braces
    (r"\boxed{\frac{3}{4} + 1}", r"\frac{3}{4} + 1"),
    (r"first \boxed{1}, then \boxed{2}", "2"),                  # last boxed wins
    (r"area is \fbox{16}", "16"),
    ("Step 1: 2+2.\n#### 4", "4"),
    ("blah\n#### 1,024", "1,024"),                              # gsm8k thousands comma
    ("The answer is 150.", "150"),
    ("The final answer is **8**.", "8"),
    ("final answer: 3.14", "3.14"),
    ("Therefore the answer is x = 9", "9"),
    ("computation gives 6 + 1 = 7", "7"),                       # trailing equation
    ("we counted 3 apples and 4 oranges", "4"),                 # last-number fallback
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_extract_answer_none_when_no_number():
    assert extract_answer("I cannot solve this problem.") is None
    assert extract_answer("") is None
    assert extract_answer(None) is None


@pytest.mark.parametrize("text,expected", [
    (r"\boxed{42}", "42"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{\sqrt{x^2+1}}", r"\sqrt{x^2+1}"),
    (r"no box here", None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("$42$", "42"),
    (r"\$42", "42"),
    ("42 km", "42"),
    ("150 km/h", "150"),
    ("30 degrees", "30"),
    (r"\text{John}", "John"),
    (r"\left(3\right)", "(3)"),
    ("1,000,000", "1000000"),
    ("42.", "42"),
    (r"\dfrac{1}{2}", r"\frac{1}{2}"),
    ("  7  ", "7"),
])
def test_normalise(raw, expected):
    assert normalise(raw) == expected


# --------------------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold", [
    ("42", "42"),
    ("42 km", "42"),
    ("$42$", "42"),
    ("1/2", "0.5"),
    ("0.5", "1/2"),
    (r"\frac{1}{2}", "0.5"),
    (r"\frac{1}{2}", "1/2"),
    ("1,000", "1000"),
    ("-7", "-7"),
    ("3.0", "3"),
    ("0.3333", "0.33333333"),    # within numeric tolerance
    ("50%", "50"),
])
def test_equivalent_true(pred, gold):
    assert answers_equivalent(pred, gold)


@pytest.mark.parametrize("pred,gold", [
    ("42", "43"),
    ("1/2", "1/3"),
    ("-7", "7"),
    ("100", "10"),
    ("x", "y"),
    (None, "5"),
    ("apple", "banana"),
])
def test_equivalent_false(pred, gold):
    assert not answers_equivalent(pred, gold)


@pytest.mark.skipif(not HAVE_SYMPY, reason="sympy not installed")
@pytest.mark.parametrize("pred,gold", [
    ("2/4", "1/2"),
    ("(x+1)^2", "x^2+2*x+1"),
    ("3*x + 2*x", "5*x"),
    (r"\frac{2}{4}", r"\frac{1}{2}"),
])
def test_equivalent_symbolic(pred, gold):
    assert answers_equivalent(pred, gold)


# --------------------------------------------------------------------------------------
# End-to-end reward
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("completion,gold", [
    (r"... hence \boxed{42}.", "42"),
    ("The answer is **150 km**.", "150"),
    ("Step 1...\n#### 18", "18"),
    (r"Thus x = \boxed{\frac{1}{2}}.", "0.5"),
    ("We find the value equals 7.", "7"),
    (r"so the perimeter is \boxed{12} cm", "12"),
])
def test_reward_correct(completion, gold):
    assert math_reward(completion, gold) == 1.0
    assert is_correct(completion, gold)


@pytest.mark.parametrize("completion,gold", [
    (r"We get \boxed{7}.", "8"),
    ("#### 5", "6"),
    ("The answer is 100.", "10"),
])
def test_reward_incorrect(completion, gold):
    assert math_reward(completion, gold) == -1.0
    assert not is_correct(completion, gold)


def test_reward_no_answer():
    assert math_reward("I am not sure.", "5") == -1.0
    # custom no_answer penalty is honoured and distinct from "incorrect"
    assert math_reward("no idea", "5", no_answer=-1.5) == -1.5


def test_reward_scale_is_configurable():
    assert math_reward(r"\boxed{3}", "3", correct=2.0) == 2.0
    assert math_reward(r"\boxed{3}", "4", incorrect=0.0) == 0.0
