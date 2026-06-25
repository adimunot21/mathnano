"""FastAPI app for the MathNano product: solve / chat / streaming chat + a web UI.

`create_app(solver)` is a factory so tests inject a dummy solver; the module-level `app` builds
the real solver from env (`build_default_solver`). Endpoints:
  GET  /health        -> liveness + model name
  POST /solve         -> {answer, solution} for one problem
  POST /chat          -> {answer, reply} for a messages[] conversation
  POST /chat/stream   -> text/event-stream of solution chunks (SSE)
  GET  /              -> the chat UI (serve/ui/index.html)
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from serve.inference import MathSolver, build_default_solver

UI_PATH = os.path.join(os.path.dirname(__file__), "ui", "index.html")


class SolveRequest(BaseModel):
    problem: str
    temperature: float = 0.0


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.0


def create_app(solver: Optional[MathSolver] = None) -> FastAPI:
    solver = solver or build_default_solver()
    app = FastAPI(title="MathNano", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": solver.model_name}

    @app.post("/solve")
    def solve(req: SolveRequest):
        s = solver.solve(req.problem, temperature=req.temperature)
        return {"answer": s.answer, "solution": s.solution}

    @app.post("/chat")
    def chat(req: ChatRequest):
        s = solver.chat([m.model_dump() for m in req.messages], temperature=req.temperature)
        return {"answer": s.answer, "reply": s.solution}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        user_turns = [m.content for m in req.messages if m.role == "user"]
        problem = user_turns[-1] if user_turns else ""

        async def event_gen():
            for piece in solver.stream(problem, temperature=req.temperature):
                yield {"event": "chunk", "data": piece}
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_gen())

    @app.get("/")
    def index():
        if os.path.exists(UI_PATH):
            return FileResponse(UI_PATH)
        return JSONResponse({"message": "MathNano API. UI not found; POST /solve."})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
