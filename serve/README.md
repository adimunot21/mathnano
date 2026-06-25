# MathNano — inference server

A small FastAPI app + web chat UI that serves the MathNano model as a step-by-step math solver.
Backend-agnostic: it wraps the same `Generator` the eval harness uses, so **the deployed model is
exactly the one we benchmarked**.

## Run locally

```bash
pip install -r requirements.txt
pip install torch transformers accelerate peft       # inference stack

# Dev (no model — dummy backend, UI works for layout):
python -m uvicorn serve.api:app --port 8000

# Real model (the Track B product):
MATHNANO_MODEL=Qwen/Qwen2.5-1.5B-Instruct python -m uvicorn serve.api:app --port 8000

# Our fine-tuned LoRA adapter on top of the base:
MATHNANO_MODEL=Qwen/Qwen2.5-1.5B MATHNANO_ADAPTER=track_b/outputs/grpo \
  python -m uvicorn serve.api:app --port 8000
```

Open http://localhost:8000 for the chat UI.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, model}` |
| POST | `/solve` | `{problem, temperature?}` | `{answer, solution}` |
| POST | `/chat` | `{messages:[{role,content}], temperature?}` | `{answer, reply}` |
| POST | `/chat/stream` | same as `/chat` | SSE stream of solution chunks |
| GET | `/` | — | chat UI |

`answer` is the extracted, machine-checkable final answer (via the shared reward's
`extract_answer`) — handy for building things on top (auto-grading, API consumers, agents).

## Docker

```bash
docker build -f serve/Dockerfile -t mathnano .
docker run -e MATHNANO_MODEL=Qwen/Qwen2.5-1.5B-Instruct -p 8000:8000 mathnano
```

## Env vars

- `MATHNANO_MODEL` — HF model id (unset → dummy backend, dev mode).
- `MATHNANO_ADAPTER` — path/id of a LoRA adapter to load on top of the base.
- `MATHNANO_4BIT=1` — load the base in 4-bit (fits larger bases on 24 GB).
- `PORT` — server port (default 8000).

## Tests

```bash
pytest serve/tests -q     # plumbing verified with a dummy backend, no ML deps
```
