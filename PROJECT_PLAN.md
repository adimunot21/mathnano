# MathNano — Project Plan

A 7-phase project to build a mathematical reasoning LLM from scratch.
Each phase has a technical goal AND a learning objective.

Total compute budget: £25.
Expected duration: 4–8 weeks depending on pace.

---

## ⚠️ Revised plan (2026-06-25) — read this before the phases below

The authoritative, current plan lives in the approved plan file and in
`experiments/nanochat_reading_notes.md`. The phase prose below is kept for its learning
objectives, but these structural changes override it:

**Two tracks + a product.**
- **Track A — from scratch (learning):** nanochat ~200M (depth=16, fallback depth=12).
  Pretrain on MathPile → SFT (extend `tasks/`) → GRPO (reuse nanochat's GSM8K `chat_rl.py`,
  add a `tasks/math.py`). This is the deep-understanding artifact.
- **Track B — capable product (quality):** SFT + GRPO (LoRA, TRL + vLLM rollouts) on
  **Qwen2.5-1.5B**. This is the model the product ships. Lives in `track_b/`.
- **Product:** chat web UI + clean inference API in `serve/` (nanochat already ships
  `chat_web.py` + `ui.html` + `chat_cli.py` we can build on).

**Command/flag corrections** (the examples in Phases 3–6 below are obsolete):
- Scripts are `scripts/tok_train.py`, `scripts/base_train.py`, `scripts/chat_sft.py`,
  `scripts/chat_rl.py`. Flags use **dashes** and there is **no `--data_path`/`--hf_repo`**.
- Real smoke test: `python -m scripts.base_train --depth=8 --max-seq-len=1024
  --device-batch-size=4 --total-batch-size=8192 --num-iterations=20 --core-metric-every=-1`.
- Init loss ≈ **10.4** (vocab 32,768), primary metric **bits-per-byte**, not 9.7.
- Data:param ratio default is **12**; pass `--target-param-data-ratio=20` for Chinchilla.
- Feed MathPile by setting `NANOCHAT_BASE_DIR` and placing parquet shards in
  `base_data_climbmix/` — **not** via a `.bin` file.

**Budget priority rule:** £25 is tight for two tracks on one 4090. If budget runs low,
**complete Track B (the product) first**; Track A's pretrain may be shortened/stopped.

**Realistic targets** at this scale (revise the optimistic tables below): Track A is mostly
educational; Track B (Qwen2.5-1.5B base) is where genuinely useful GSM8K/MATH numbers come from.

---

## Phase 0 — Environment Setup and nanochat Deep-Read

**Duration**: 2–3 days, no GPU needed  
**Compute cost**: £0

### Goal

Set up the development environment and read every line of nanochat before
touching it. You cannot modify a codebase you don't understand.

### Tasks

- [ ] Install dependencies: Python 3.11+, PyTorch 2.x, CUDA toolkit
- [ ] Clone nanochat: `git clone https://github.com/karpathy/nanochat`
- [ ] Clone this repo: set up the mathnano/ structure from CLAUDE.md
- [ ] Read `nanochat/model.py` completely — no running, just reading
- [ ] Read `nanochat/tokenizer.py` — understand how BPE is implemented
- [ ] Read `scripts/base_train.py` — trace the training loop step by step
- [ ] Read `scripts/sft_train.py` — understand how SFT differs from pretraining
- [ ] Read `scripts/grpo_train.py` — understand how GRPO works at a high level
- [ ] Read `speedrun.sh` — understand how all stages chain together
- [ ] Run the nanochat tiny test: `python -m nanochat.test_model` (if present)
- [ ] Write `experiments/nanochat_reading_notes.md` (see deliverable)

### Learning Objective

By the end of Phase 0, you should be able to answer:
- Where is the transformer architecture defined?
- What does the training loop do between calling forward() and backward()?
- What does the `--depth` parameter control and how does it scale dimensions?
- How does nanochat format data for pretraining vs SFT vs GRPO?
- Where does the Muon optimizer get applied vs AdamW?

### Deliverable

`experiments/nanochat_reading_notes.md` — a file with your own answers to
the questions above, written in plain language. This is your baseline
understanding. You will update it as you learn more throughout the project.

---

## Phase 1 — Data Pipeline

**Duration**: 3–4 days, CPU only  
**Compute cost**: £0–1 (possible cheap CPU pod for large downloads)

### Goal

Download MathPile, understand what's in it, and process it into the binary
format nanochat expects. Understand what "tokens" look like for mathematical text.

### Understanding MathPile

MathPile is the pretraining dataset. It contains:
- **arXiv papers** (~85% of the dataset): research math papers in LaTeX
- **Textbooks** (~5%): undergraduate and graduate-level math books
- **StackExchange** (~5%): math Q&A, concrete problem solving
- **ProofWiki** (~3%): formal mathematical proofs
- **Wikipedia** (~2%): encyclopedic math definitions and theorems

Total: 9.5 billion tokens. This is almost exactly nanochat's optimal pretraining
budget for depth=16. The alignment is not a coincidence — it's why MathPile was chosen.

**Why this mix matters**: arXiv papers teach the model the *language* of mathematics —
notation, proof structure, theorem/lemma/corollary patterns. Textbooks and StackExchange
teach more elementary reasoning. The mix gives the model breadth.

### Tasks

- [ ] Download MathPile:
  ```bash
  from datasets import load_dataset
  ds = load_dataset("EleutherAI/mathpile", split="train")
  ```
- [ ] Write `mathnano/data/inspect_data.py`:
  - Sample 100 documents from each source (arXiv, textbooks, StackExchange)
  - Print them — read them. What does raw math text look like?
  - Count unique characters. How many LaTeX commands appear?
  - Plot token length distribution
- [ ] Train the BPE tokenizer on MathPile:
  - nanochat includes a tokenizer training script
  - Vocabulary size: 16,384 (nanochat default for small models)
  - Understand what merges the BPE makes — does it create tokens for common LaTeX?
- [ ] Write `mathnano/data/prepare_mathpile.py`:
  - Tokenise all of MathPile
  - Save as binary file (uint16 token IDs)
  - Create train/val split: 99% train, 1% val
- [ ] Prepare SFT datasets (can reuse main tokenizer):
  - Download OpenMathInstruct-2: `load_dataset("nvidia/OpenMathInstruct-2")`
  - Download GSM8K: `load_dataset("gsm8k", "main")`
  - Download MATH: `load_dataset("hendrycks/competition_math")`
  - Write `mathnano/data/prepare_sft.py` to format all three into nanochat chat format

### The nanochat Data Format

For pretraining: a flat binary file of token IDs. The training loop slices windows.

For SFT: one conversation per line in JSONL:
```json
{"messages": [
  {"role": "user", "content": "What is 2 + 2?"},
  {"role": "assistant", "content": "Step 1: Add the numbers.\n2 + 2 = 4\nThe answer is 4."}
]}
```

For GRPO: problems with verified answers:
```json
{"problem": "If x + 3 = 7, what is x?", "answer": "4"}
```

### Learning Objective

By the end of Phase 1, you should be able to answer:
- What does a tokeniser do to `\frac{d}{dx} x^2 = 2x`? How many tokens?
- What's the difference between byte-pair encoding and word-level tokenisation?
- Why does vocabulary size matter? What's the trade-off between 8K and 64K vocab?
- What does a BF16 binary file of tokens look like in memory?

### Deliverable

Processed data in `mathnano/data/processed/`:
- `mathpile_train.bin` — tokenised pretraining data
- `mathpile_val.bin` — validation split
- `sft_combined.jsonl` — SFT data
- `grpo_problems.jsonl` — GRPO problems with answers
- `data_statistics.md` — document counts, token counts, sample documents

---

## Phase 2 — Architecture Deep-Dive

**Duration**: 4–6 days, no GPU needed  
**Compute cost**: £0

### Goal

Understand every component of the model by reading nanochat's code and
implementing each piece from scratch in a notebook. Read ARCHITECTURE.md
alongside this phase — it is the reference document.

### Why Implement From Scratch?

There is a difference between reading code and writing code. Writing forces
you to confront every implicit assumption. You will not understand RoPE until
you have computed the rotation matrix by hand for a 3-token sequence.

The implementations in notebooks do not need to be efficient or complete —
they are for understanding. nanochat's implementation is what we actually train.

### Tasks (follow ARCHITECTURE.md section by section)

**Tokenisation**
- [ ] Open `notebooks/01_tokenizer_deep_dive.ipynb`
- [ ] Manually trace a BPE merge operation on a tiny vocabulary
- [ ] Tokenise 10 math problems by hand (with the trained tokenizer)
- [ ] Answer: what tokens does the model use for `\sqrt{x}`? For `\leq`?

**Embeddings**
- [ ] What is an embedding table? Why is it different from a one-hot encoding?
- [ ] What is "weight tying" between the input embedding and the output projection?
- [ ] Why does the embedding table have shape (vocab_size, d_model)?

**RoPE**
- [ ] Open `notebooks/02_attention_from_scratch.ipynb`
- [ ] Implement the rotation formula for a single (x_2i, x_2i+1) pair
- [ ] Compute rotations for positions 0, 1, 2, 3 and dimension indices 0, 1
- [ ] Verify: the dot product of query at pos 3 and key at pos 1 depends only
  on (3 - 1) = 2, not on 3 or 1 separately
- [ ] Plot the rotation angles across positions and dimensions — what pattern do you see?

**RMSNorm vs LayerNorm**
- [ ] Implement both in 5 lines each
- [ ] Time them on a (32, 512, 1024) tensor — how much faster is RMSNorm?
- [ ] Run both on a vector with large outliers — what happens?

**Multi-Head Attention (MHA)**
- [ ] Implement scaled dot-product attention from scratch (no library calls)
- [ ] Implement the full MHA module: Q/K/V projections, split into heads, attend, concat
- [ ] Implement the causal mask — why must token 3 not attend to token 4?
- [ ] Check: what is the shape of the attention weight matrix for (batch=2, heads=8,
  seq=32, head_dim=64)? How much memory does it use?

**Grouped Query Attention (GQA)**
- [ ] Modify your MHA to have n_kv_heads < n_heads
- [ ] Implement the repeat_interleave trick to expand K and V
- [ ] With n_heads=8, n_kv_heads=2, seq=512, d_model=1024:
  - What is the MHA KV cache size in bytes (float16)?
  - What is the GQA KV cache size?
  - What is the reduction factor?

**SwiGLU FFN**
- [ ] Implement the standard ReLU FFN (2 matrices)
- [ ] Implement SwiGLU (3 matrices, 8/3 hidden dim)
- [ ] Count parameters for each: which is larger?
- [ ] Run both on identical random input for 100 steps of a toy regression task
  — does SwiGLU converge faster?

**Full Transformer Block**
- [ ] Assemble: RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual
- [ ] This is one "layer". nanochat depth=16 stacks 16 of these.
- [ ] Trace a single token through the block, tracking the tensor shape at each step

**Language Model Head**
- [ ] The final layer maps (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
- [ ] Weight tying: this projection reuses the embedding table weights
- [ ] The cross-entropy loss: what does it measure? Why does lower = better?

### Learning Objective

By the end of Phase 2, you should be able to:
- Implement attention from scratch without looking at reference code
- Explain why each of RoPE, RMSNorm, SwiGLU, GQA was chosen
- Give the tensor shapes at every step of the forward pass
- Explain the KV cache: why it exists, what it stores, how GQA shrinks it
- Read nanochat's model.py and understand every line

### Deliverable

`notebooks/02_attention_from_scratch.ipynb` — working implementations of
each component with shape annotations and the explanations from above.

---

## Phase 3 — Training Infrastructure

**Duration**: 2–3 days, optional short GPU test  
**Compute cost**: ~£0.50–1

### Goal

Understand nanochat's training loop, write configs for MathPile, and run a
smoke test at depth=8 to verify everything works before spending real money.

### The Training Loop

Understanding this loop is as important as understanding the architecture.
Training loop steps, for each batch:

1. **Forward pass**: tokens → model → logits (shape: batch × seq_len × vocab_size)
2. **Loss**: cross-entropy between logits and the next token (shifted input)
3. **Backward pass**: compute gradients via automatic differentiation
4. **Gradient accumulation**: wait N micro-batches before updating weights
   (simulates larger batch size without more memory)
5. **Gradient clipping**: cap gradient norm to prevent explosive updates
6. **Optimiser step**: Muon updates 2D weights, AdamW updates 1D weights
7. **Logging**: loss, learning rate, MFU (model FLOP utilisation)
8. **Checkpointing**: save model and optimiser state periodically

### What is Gradient Accumulation?

Training with large batch sizes stabilises learning but requires proportionally
more memory. Gradient accumulation is a workaround: run N forward/backward passes
on small micro-batches, accumulate the gradients without updating weights, then
do a single optimiser step. The effect is identical to one pass on a batch of
size N × micro_batch_size.

nanochat targets a global batch size of ~500K tokens. With a micro batch of
8 sequences × 1024 tokens = 8192 tokens, you need 500K / 8192 ≈ 61 accumulation
steps. This is set automatically by nanochat based on --depth.

### What is MFU?

Model FLOP Utilisation. The theoretical peak FLOPS of an RTX 4090 in BF16 is
165.2 TFLOPS. MFU measures what fraction of that your training run actually
achieves. A well-tuned nanochat run hits ~40–50% MFU on 4090. If you see <20%,
something is wrong (usually data loading bottleneck or wrong batch size).

### What is BF16?

Brain Float 16. A 16-bit floating point format with the same exponent range as
FP32 but fewer mantissa bits. It handles the large dynamic range needed for neural
networks (unlike FP16) while using half the memory. All modern LLM training
uses BF16. PyTorch's `torch.bfloat16` is what you'll set in the config.

### Tasks

- [ ] Write `mathnano/config/pretrain_d8.sh`:
  ```bash
  torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    --depth=8 \
    --data_path=mathnano/data/processed/mathpile_train.bin \
    --val_data_path=mathnano/data/processed/mathpile_val.bin \
    --output_dir=experiments/runs/d8_smoke \
    --max_steps=200 \
    --device_batch_size=8
  ```
- [ ] Write `mathnano/config/pretrain_d16.sh` — same but depth=16, max_steps=full
- [ ] Run the depth=8 smoke test for 200 steps locally or on RunPod
  - What does the loss start at? (Expect ~ln(vocab_size) ≈ 9.7 for 16K vocab)
  - Does it decrease? (It should, within the first 50 steps)
  - What is the MFU?
- [ ] Read nanochat's checkpoint format — what is saved in each checkpoint?
- [ ] Write a script to push a checkpoint to HuggingFace Hub
- [ ] Understand the learning rate schedule: linear warmup + cosine decay

### Learning Objective

By the end of Phase 3, you should be able to:
- Explain gradient accumulation without looking it up
- Interpret a training log: what does a good loss curve look like?
- Explain what MFU is and why it matters
- Configure a training run from scratch
- Know how to restart from a checkpoint if a pod is preempted

### Deliverable

Working configs in `mathnano/config/`, smoke test log in `experiments/logs/`,
a brief `experiments/smoke_test_notes.md` interpreting the run.

---

## Phase 4 — Pretraining on MathPile

**Duration**: 15–25 GPU hours (RunPod)  
**Compute cost**: ~£6–10

### Goal

Run the full depth=16 pretraining on MathPile. Monitor the run. Generate
completions mid-training to watch the model's understanding develop.

### What Happens During Pretraining

The model starts as random noise. At step 0, loss is ~ln(vocab_size) ≈ 9.7
(uniform distribution over all tokens). Over training:

- **Steps 0–500**: loss drops rapidly as the model learns word frequencies
- **Steps 500–5000**: loss drops more slowly as the model learns syntax and structure
- **Steps 5000+**: slow steady improvement as the model learns meaning and reasoning

By the end of pretraining on MathPile, the model will:
- Complete mathematical sentences coherently
- Continue proofs in a reasonable style
- Generate plausible-looking LaTeX
- NOT reliably solve problems or follow instructions (that comes in SFT)

### RunPod Setup

See RUNPOD.md for full setup guide. Quick summary:
1. Choose RTX 4090 spot, Community Cloud
2. Use the PyTorch template
3. SSH in, clone this repo
4. Copy processed data (or download directly from HuggingFace on the pod)
5. Start training with tmux so it survives SSH disconnection

### Tasks

- [ ] Spin up RTX 4090 spot pod on RunPod
- [ ] Verify GPU: `nvidia-smi` should show one RTX 4090 with ~24GB VRAM
- [ ] Start training: `bash mathnano/config/pretrain_d16.sh`
- [ ] Watch the first 100 steps closely — loss should decrease
- [ ] Check MFU — aim for >35%. If lower, investigate data loading
- [ ] At step 1000, generate a few completions:
  ```python
  prompt = "Theorem 1.1. For all integers n ≥ 1,"
  # What does the model predict next?
  ```
- [ ] At step 5000, try a harder prompt:
  ```python
  prompt = "To solve the equation 2x + 5 = 13, we first"
  ```
- [ ] Checkpoint every 30 minutes, push to HuggingFace
- [ ] If pod is preempted, resume from latest checkpoint
- [ ] Final evaluation: perplexity on validation set

### What to Expect Mid-Training

Step 500 completions will be incoherent but grammatical — real words in
math-like arrangements. Step 5000 completions will have LaTeX structure.
Step 20000+ completions will look like real (though possibly wrong) math text.

### Learning Objective

By the end of Phase 4, you should be able to:
- Interpret a training loss curve and identify common failure modes
- Understand what "perplexity" measures (exp(loss)) and why it matters
- Explain why pretraining alone doesn't give you a chatbot
- Describe what the model "knows" at the end of pretraining

### Deliverable

Pretrained checkpoint on HuggingFace (`your-username/mathnano-pretrain-d16`),
training log in `experiments/logs/`, loss curve plot in `experiments/`.

---

## Phase 5 — Supervised Fine-Tuning (SFT)

**Duration**: 3–6 GPU hours (RunPod)  
**Compute cost**: ~£1–2.50

### Goal

Convert the pretrained model from "math text completer" to "math problem solver
that follows instructions". This is the step that makes it a chatbot.

### What SFT Does

The pretrained model predicts the next token in any mathematical text. It has
no concept of "question" and "answer" — it treats them symmetrically. SFT
teaches it a particular behaviour: given a question in a specific format,
produce a step-by-step solution ending with a boxed answer.

During SFT, we only compute loss on the **assistant's** tokens. The user turn
is given to the model as context (the prompt) but its tokens are masked out
of the loss. This prevents the model from wasting capacity learning to predict
the questions.

### The Chat Format

nanochat uses this format (special tokens defined in the tokenizer):
```
<|user|>
A train travels 60 km/h for 2.5 hours. How far does it travel?
<|endoftext|>
<|assistant|>
Step 1: Identify the formula.
Distance = Speed × Time

Step 2: Substitute the values.
Distance = 60 km/h × 2.5 h = 150 km

The answer is **150 km**.
<|endoftext|>
```

The model is trained to predict every token after `<|assistant|>`. The tokens
before it (the user turn) are provided as context but are not in the loss.

### The Data Mix

We use three datasets:
- **OpenMathInstruct-2**: 7.5M examples, high quality, chain-of-thought solutions
- **GSM8K train**: 7473 examples, grade school word problems
- **MATH train**: 7500 examples, competition math with worked solutions

Mix ratio: 80% OpenMathInstruct-2, 10% GSM8K, 10% MATH.

Why mostly OpenMathInstruct-2? It's large and high-quality. GSM8K and MATH
are included to ensure the model is exposed to the exact benchmark format.

### Tasks

- [ ] Format SFT data into nanochat chat format (see prepare_sft.py from Phase 1)
- [ ] Understand the learning rate: SFT uses a much lower LR than pretraining
  (typically 10–50× lower). Why? Because you want to adjust behaviour without
  forgetting the knowledge from pretraining.
- [ ] Run SFT: `bash mathnano/config/sft_d16.sh`
- [ ] After training, evaluate on GSM8K test set with `eval_gsm8k.py`
- [ ] Compare: ask the pretrained model and SFT model the same GSM8K question.
  The difference is stark — document it.

### Catastrophic Forgetting

There is a real risk: fine-tuning on a narrow dataset can make the model
"forget" what it learned in pretraining. Signs include:
- Loss on the original MathPile validation set increasing significantly
- The model refusing to generate coherent mathematical text on non-problem inputs
- Degraded perplexity on held-out math papers

To mitigate: use a low learning rate, few epochs, and monitor validation loss
on MathPile (not just SFT loss).

### Learning Objective

By the end of Phase 5, you should be able to:
- Explain what SFT does and why pretraining alone is insufficient
- Explain why we only backpropagate on assistant tokens
- Describe the catastrophic forgetting problem and how to detect it
- Interpret GSM8K accuracy as a metric

### Deliverable

SFT checkpoint on HuggingFace, GSM8K baseline score, before/after comparison
document in `experiments/phase5_comparison.md`.

---

## Phase 6 — GRPO with Verifiable Rewards

**Duration**: 6–12 GPU hours (RunPod)  
**Compute cost**: ~£2.50–5

### Goal

Apply reinforcement learning to teach the model to get correct answers, not
just produce well-formatted solutions. This is the most technically novel part
of the project.

### What GRPO Is

GRPO stands for Group Relative Policy Optimisation. It is the algorithm used in
DeepSeek-R1 and is nanochat's built-in RL method.

For each training problem:
1. Sample G completions from the current model (G = "group size", typically 4–8)
2. Evaluate each completion with the reward function → rewards r₁, r₂, ..., rG
3. Compute the relative reward for each: ρᵢ = rᵢ - mean(r) / std(r)
4. Update the model to increase probability of high-relative-reward completions

**Why relative rewards**: instead of needing an absolute reward scale, we just
ask "which of these G solutions was better?" This is much more stable than
policy gradient methods that require careful reward normalisation.

**Why G > 1**: generating multiple completions per problem lets the model
explore different solution strategies. If one works (reward +1) and others
don't (reward -1), the model learns to favour the successful approach.

### The Math Reward Function

This is in `mathnano/rewards/math_reward.py`. It does two things:
1. Extract the final numerical answer from the model's output
2. Compare it to the ground truth

Extracting the answer is the hard part. Models write answers in various formats:
- "The answer is **42**."
- "Therefore, x = 42."
- "\\boxed{42}"
- "So the total is 42 km."

You need a robust extractor. The standard approach:
```python
import re

def extract_answer(text: str) -> str | None:
    # Look for \boxed{...} first (competition math convention)
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    
    # Look for "the answer is X" pattern
    answer_pattern = re.search(
        r'(?:the answer is|= |equals )\**(-?\d+(?:\.\d+)?)\**',
        text.lower()
    )
    if answer_pattern:
        return answer_pattern.group(1)
    
    # Fall back to the last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None

def math_reward(completion: str, ground_truth: str) -> float:
    extracted = extract_answer(completion)
    if extracted is None:
        return -1.0  # No answer found
    
    try:
        # Numerical comparison with tolerance
        if abs(float(extracted) - float(ground_truth)) < 1e-6:
            return 1.0
    except ValueError:
        # String comparison for symbolic answers
        if extracted.strip() == ground_truth.strip():
            return 1.0
    
    return -1.0
```

### Tasks

- [ ] Write and thoroughly test `mathnano/rewards/math_reward.py`
  - Test on 50 manually crafted examples before using in training
  - Edge cases: fractions, negative numbers, units (e.g. "42 km"), percentages
- [ ] Understand GRPO's hyperparameters:
  - `group_size G`: how many completions per problem (start with 4)
  - `clip_ratio ε`: PPO-style clipping (nanochat default: 0.2)
  - `kl_weight`: penalty for diverging too far from the SFT model
- [ ] Run GRPO: `bash mathnano/config/grpo_d16.sh`
- [ ] Monitor: reward signal (should trend positive over time), KL divergence
- [ ] Evaluate: GSM8K before vs after GRPO
- [ ] Look at 10 specific problems that the model got wrong after SFT but
  right after GRPO. What changed?

### The KL Divergence Constraint

GRPO includes a penalty for the policy diverging too far from the SFT model
(the "reference model"). This prevents reward hacking: if the reward function
has loopholes, an unconstrained RL agent will exploit them instead of actually
solving math problems. The KL penalty keeps the model from drifting to bizarre
output distributions.

### Learning Objective

By the end of Phase 6, you should be able to:
- Explain GRPO and why it's better than naive policy gradient
- Describe what reward hacking is and how KL constraints prevent it
- Quantify how much GRPO improved accuracy over SFT
- Explain why verifiable rewards are a special case (no reward model needed)
- Compare to RLHF: what's different when you have ground truth vs human preferences?

### Deliverable

GRPO checkpoint on HuggingFace, before/after evaluation report,
analysis of 10 improved responses in `experiments/phase6_analysis.md`.

---

## Phase 7 — Evaluation and Analysis

**Duration**: 1–2 days, minimal GPU  
**Compute cost**: ~£0.50–1

### Goal

Rigorously evaluate the final model. Understand what it learned, what it
didn't, and why. Write a model card.

### Tasks

- [ ] Run full GSM8K evaluation (1319 test problems)
- [ ] Run MATH evaluation (5000 test problems, 5 difficulty levels)
- [ ] Try AMC 2023/2024 problems (hard, expect ~0–5%)
- [ ] Compile the progression table:

  | Stage        | GSM8K  | MATH Level 1 | MATH Level 5 |
  |--------------|--------|--------------|--------------|
  | After pretrain | ~5%  | ~1%          | ~0%          |
  | After SFT    | ~25%   | ~8%          | ~1%          |
  | After GRPO   | ~40%   | ~15%         | ~3%          |

  (These are estimates — your actual numbers may differ.)

- [ ] Failure analysis: pick 20 problems the final model gets wrong.
  Categorise the error types:
  - Arithmetic errors (correct reasoning, wrong calculation)
  - Reasoning errors (wrong approach from the start)
  - Format errors (right answer, not detected by extractor)
  - Knowledge gaps (requires facts the model doesn't have)
- [ ] Write a model card for HuggingFace
- [ ] Push final model: `your-username/mathnano-d16-final`

### The Model Card

Document in the HuggingFace model card:
- What the model is and how it was trained
- Training data and its limitations
- Benchmark scores
- Failure modes
- How to use it (example inference code)
- What it should NOT be used for

### Learning Objective

By the end of Phase 7, you should be able to:
- Interpret benchmark numbers in context (what do these scores mean relative
  to model size and training compute?)
- Do principled failure analysis
- Describe the limitations of your model honestly
- Understand the evaluation gap: why do models perform differently on
  benchmarks vs real-world use?

### Deliverable

Full evaluation report, model card, final checkpoint on HuggingFace.

---

## Milestone Summary

| Phase | Milestone test                                                |
|-------|---------------------------------------------------------------|
| 0     | Can describe what every nanochat file does from memory        |
| 1     | Tokenised MathPile, visualised 20 examples, data stats done   |
| 2     | Implemented attention from scratch, can explain RoPE + GQA   |
| 3     | Smoke test (depth=8) passes, loss decreases in 100 steps     |
| 4     | Pretrained model generates coherent mathematical text         |
| 5     | Model solves >20% of GSM8K problems after SFT                |
| 6     | GRPO improves GSM8K accuracy vs SFT baseline (any improvement) |
| 7     | Full evaluation complete, model public on HuggingFace         |

---

## Budget Summary

| Phase | Activity                          | GPU hrs | Cost (RTX 4090 spot) |
|-------|-----------------------------------|---------|----------------------|
| 0–2   | Reading, notebooks, no GPU        | 0       | £0                   |
| 3     | Smoke test ×2 (depth=8, 200 steps)| 1–2     | ~£0.50–1             |
| 4     | Test run (depth=8, full data)     | 1–2     | ~£0.50–1             |
| 4     | Main pretrain (depth=16)          | 15–25   | ~£6–10               |
| 5     | SFT                               | 3–6     | ~£1–2.50             |
| 6     | GRPO                              | 6–12    | ~£2.50–5             |
| 7     | Evaluation                        | 0.5–1   | ~£0.20–0.50          |
| —     | Buffer (reruns, debugging)        | 5–15    | ~£2–6                |
| **Total** |                               | **32–64** | **~£13–26**        |

Budget ceiling: £25. Do not exceed without reviewing this table.

The main risk is pretraining taking longer than expected due to data loading
bottlenecks or lower MFU than expected. If this happens, reduce depth to 12
(~120M params) rather than cutting training short.
