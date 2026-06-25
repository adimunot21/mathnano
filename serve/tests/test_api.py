"""API tests with a dummy solver — verify product plumbing with no ML deps.

Run: `pytest serve/tests -q`. Uses FastAPI TestClient (needs httpx).
"""
from fastapi.testclient import TestClient

from mathnano.eval.runner import DummyGenerator
from serve.api import create_app
from serve.inference import MathSolver


def make_client(oracle=None):
    solver = MathSolver(DummyGenerator(oracle=oracle), model_name="test")
    return TestClient(create_app(solver))


def test_health():
    c = make_client()
    r = c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model": "test"}


def test_solve_extracts_answer():
    c = make_client(oracle=lambda p: "Step 1... so \\boxed{42}.")
    r = c.post("/solve", json={"problem": "what is 6*7?"})
    body = r.json()
    assert body["answer"] == "42"
    assert "boxed{42}" in body["solution"]


def test_chat_uses_last_user_turn():
    c = make_client(oracle=lambda p: f"answer for: {p} is \\boxed{{7}}")
    r = c.post("/chat", json={"messages": [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "3+4?"},
    ]})
    body = r.json()
    assert body["answer"] == "7"
    assert "3+4?" in body["reply"]  # collapsed to the latest user question


def test_chat_stream_reassembles_full_text():
    full = "The answer is \\boxed{15} after some steps." * 3
    c = make_client(oracle=lambda p: full)
    with c.stream("POST", "/chat/stream",
                  json={"messages": [{"role": "user", "content": "q"}]}) as r:
        assert r.status_code == 200
        chunks = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                chunks.append(line[len("data:"):].lstrip())
        # reassembled stream contains the boxed answer
        assert "boxed{15}" in "".join(chunks)


def test_index_served():
    c = make_client()
    r = c.get("/")
    assert r.status_code == 200
    assert "MathNano" in r.text
