# MathNano inference server.
# Builds a CPU image by default (works with any HF model, slower). For the real Qwen2.5-1.5B
# product, run on a CUDA host and install a CUDA torch wheel (see the commented line below).
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; transformers needs no compiler for inference.
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Shared deps (FastAPI, datasets-free at runtime but small) + inference stack.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch transformers accelerate peft
# GPU variant: replace the torch line above with, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124

# App code (mathnano package is imported by serve.inference / serve.api).
COPY mathnano ./mathnano
COPY serve ./serve

EXPOSE 8000
ENV PORT=8000
# Configure the model at run time:
#   docker run -e MATHNANO_MODEL=Qwen/Qwen2.5-1.5B-Instruct -p 8000:8000 mathnano
# With our LoRA adapter:
#   -e MATHNANO_MODEL=Qwen/Qwen2.5-1.5B -e MATHNANO_ADAPTER=/app/adapter -v $PWD/adapter:/app/adapter
CMD ["python", "-m", "uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
