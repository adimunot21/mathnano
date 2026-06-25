"""Tests for the eval engine, using DummyGenerator (no ML deps). `pytest mathnano/eval -q`."""
from mathnano.eval.runner import DummyGenerator, EvalRow, evaluate

ROWS = [
    EvalRow(problem="2+2?", answer="4", level=1),
    EvalRow(problem="3*3?", answer="9", level=2),
    EvalRow(problem="10-1?", answer="9", level=2),
]
GOLD = {"2+2?": "4", "3*3?": "9", "10-1?": "9"}


def test_perfect_oracle_scores_100():
    gen = DummyGenerator(oracle=lambda p: f"... so \\boxed{{{GOLD[p]}}}.")
    res = evaluate(ROWS, gen, progress=False)
    assert res.correct == 3 and res.accuracy == 1.0


def test_wrong_oracle_scores_0():
    gen = DummyGenerator(oracle=lambda p: "the answer is \\boxed{-999}.")
    res = evaluate(ROWS, gen, progress=False)
    assert res.correct == 0 and res.accuracy == 0.0


def test_per_level_breakdown():
    # correct only on level-2 rows
    gen = DummyGenerator(oracle=lambda p: "\\boxed{9}")
    res = evaluate(ROWS, gen, progress=False)
    assert res.by_level[1] == (0, 1)
    assert res.by_level[2] == (2, 2)


def test_pass_at_k_any_correct_counts():
    # alternate wrong/right across the k samples; pass@k should catch the right one
    state = {"i": 0}

    def flaky(_p):
        state["i"] += 1
        return "\\boxed{9}" if state["i"] % 2 == 0 else "\\boxed{0}"

    rows = [EvalRow(problem="3*3?", answer="9")]
    res = evaluate(rows, DummyGenerator(oracle=flaky), k=4, progress=False)
    assert res.correct == 1  # at least one of the 4 samples was right


def test_records_capture_prediction():
    gen = DummyGenerator(oracle=lambda p: f"\\boxed{{{GOLD[p]}}}")
    res = evaluate(ROWS, gen, progress=False)
    assert res.records[0]["predicted"] == "4"
    assert res.records[0]["correct"] is True
