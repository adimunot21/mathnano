# MathNano — Datasets

Everything you need to know about the data: what it is, where to get it,
how to process it, and what format nanochat expects.

---

## ⚠️ Corrections (verified 2026-06-25 — these override the older text below)

- **MathPile repo is `GAIR/MathPile`** (not `EleutherAI/mathpile`). It is **gated** (accept the
  CC BY-NC-SA 4.0 license on the HF page + use an HF token) and **non-commercial** — fine for a
  portfolio project, but note it in the model card. Download via `huggingface-cli download`
  (not `load_dataset`); files are **jsonl.gz**; the source field is **`SubSet`** (not `source`).
- **GSM8K is `openai/gsm8k`** (subset `main`).
- **`hendrycks/competition_math` was DMCA'd** and is no longer reliably loadable. Use a mirror —
  verify **`nlile/hendrycks-MATH-benchmark`** (or `qwedsacf/competition_math`) in Phase 1 and
  pin whichever has the full train/test with `problem`/`solution`/`level`/`type`.
- **There is NO uint16 `.bin` pretraining pipeline.** nanochat reads **parquet shards**
  (`shard_NNNNN.parquet`, single `text` column) from `$NANOCHAT_BASE_DIR/base_data_climbmix/`
  (last shard = val) and **tokenizes on the fly**; the tokenizer (`tok_train.py`, default
  **vocab 32,768**) trains from the same `text`. So our job is: MathPile → **text parquet
  shards** in that dir, then `tok_train` + `base_train`. The `.bin`/`--data_path` recipe below
  is obsolete — see `experiments/nanochat_reading_notes.md §3`.
- **SFT/RL data is task-based, not `{"messages": …}` JSONL.** nanochat SFT mixes `tasks/`
  Conversation datasets; `chat_rl.py` GRPO is hardcoded to `tasks/gsm8k.py`. For Track A we
  extend `tasks/`; the chat-JSONL format below applies only to **Track B** (Qwen via TRL).

---

## Overview

| Stage    | Dataset              | Tokens      | Purpose                             |
|----------|----------------------|-------------|-------------------------------------|
| Pretrain | MathPile             | 9.5B        | Teach the model mathematical text   |
| SFT      | OpenMathInstruct-2   | ~7.5M pairs | Teach instruction following + CoT   |
| SFT      | GSM8K (train)        | 7,473 pairs | Grade school problems               |
| SFT      | MATH (train)         | 7,500 pairs | Competition math with solutions     |
| GRPO     | GSM8K (train)        | 7,473 probs | RL with verifiable answers          |
| GRPO     | MATH (train)         | 7,500 probs | Harder RL problems                  |
| Eval     | GSM8K (test)         | 1,319 probs | Benchmark                           |
| Eval     | MATH (test)          | 5,000 probs | Benchmark (5 difficulty levels)     |

---

## Stage 1: MathPile (Pretraining)

### What it is

MathPile is a 9.5 billion token corpus of mathematical text assembled by the
GAIR lab. It was designed specifically for pretraining language models on math.

Composition:
- arXiv papers (~85%): research mathematics, written in LaTeX
- Textbooks (~5%): undergraduate through graduate-level
- Wikipedia math articles (~2%): definitions, theorems, examples
- StackExchange (Math) (~3%): questions and answers
- ProofWiki (~3%): formal proofs in natural language
- Other (~2%): miscellaneous

The arXiv dominance means the model will learn a lot of research-level math.
This is intentional: research papers have dense reasoning chains and formal
proof structure — exactly what we want for a reasoning model.

### Download

```python
# In mathnano/data/prepare_mathpile.py
from datasets import load_dataset

# This will download ~20GB of parquet files
ds = load_dataset("EleutherAI/mathpile", split="train", streaming=False)

# Inspect the structure
print(ds.features)
# {'text': Value(dtype='string'), 'source': Value(dtype='string')}

# Look at 3 examples from each source
for source in ['arxiv', 'textbooks', 'stackexchange', 'proofwiki', 'wikipedia']:
    examples = ds.filter(lambda x: x['source'] == source).select(range(3))
    for ex in examples:
        print(f"SOURCE: {source}")
        print(ex['text'][:500])
        print("---")
```

### Tokenisation

We use nanochat's BPE tokenizer. First train it on MathPile (nanochat does this
as part of its pipeline), then use it to tokenise all documents.

nanochat trains the tokenizer with:
```bash
python -m nanochat.train_tokenizer \
  --data_path=mathnano/data/raw/mathpile \
  --vocab_size=16384 \
  --output_path=mathnano/data/tokenizer/
```

After training, tokenise and save as binary:
```python
import numpy as np
from nanochat.tokenizer import Tokenizer

tok = Tokenizer("mathnano/data/tokenizer/")

# Tokenise all documents
all_ids = []
for item in ds:
    ids = tok.encode(item['text'])
    all_ids.extend(ids)
    all_ids.append(tok.eot_id)  # End-of-document token between documents

ids_array = np.array(all_ids, dtype=np.uint16)

# Train/val split: 99% train, 1% val
split = int(0.99 * len(ids_array))
np.array(ids_array[:split]).tofile("mathnano/data/processed/mathpile_train.bin")
np.array(ids_array[split:]).tofile("mathnano/data/processed/mathpile_val.bin")

print(f"Train tokens: {split:,}")
print(f"Val tokens: {len(ids_array) - split:,}")
```

Expected sizes:
- Train: ~9.4B tokens (~18.8 GB as uint16)
- Val: ~95M tokens (~190 MB as uint16)

### What to Inspect Before Training

Run this before committing to the full tokenisation. Understanding your data is
a core skill.

```python
# mathnano/data/inspect_data.py

# 1. Sample and print 5 documents from each source
# 2. Tokenise them and print the token IDs alongside the text
# 3. Find: what token does \frac map to? \sqrt? \leq? "therefore"? "proof"?
# 4. Compute: average token length per document by source
# 5. Plot: histogram of document lengths (in tokens)
# 6. Identify: are there any documents that are just LaTeX boilerplate with no
#    actual mathematical content? (common in arXiv preambles)
```

---

## Stage 2: SFT Datasets

### OpenMathInstruct-2

NVIDIA's dataset of 7.5 million math problem-solution pairs. Solutions are
step-by-step chain-of-thought written by Llama 3.1 405B. High quality.

```python
from datasets import load_dataset
ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")

# Structure:
# {'problem': str, 'generated_solution': str, 'expected_answer': str,
#  'problem_source': str}  # gsm8k, math, amc, etc.

# Example
print(ds[0]['problem'])
# "The sum of the interior angles of a polygon with n sides is (n-2)*180 degrees.
#  How many sides does a polygon have if the sum of its interior angles is 1080?"

print(ds[0]['generated_solution'])
# "Let's denote the number of sides as n.
#  The sum of interior angles = (n-2) * 180 = 1080
#  n - 2 = 1080 / 180 = 6
#  n = 8
#  The polygon has **8** sides."
```

### GSM8K

Grade School Math dataset. 8500 problems written by human annotators.
Problems are word problems involving basic arithmetic and proportional reasoning.

```python
from datasets import load_dataset
ds = load_dataset("gsm8k", "main")
train = ds['train']  # 7473 examples
test  = ds['test']   # 1319 examples

# Structure
# {'question': str, 'answer': str}
# Note: answers in GSM8K embed the final number after "####"
# e.g. "Step 1: ...\nStep 2: ...\n#### 42"
```

### MATH

Hendrycks et al. Competition mathematics dataset. 12500 problems at 5 difficulty
levels across 7 subjects (algebra, geometry, number theory, etc.).

```python
from datasets import load_dataset
ds = load_dataset("hendrycks/competition_math")
train = ds['train']  # 7500 examples
test  = ds['test']   # 5000 examples

# Structure
# {'problem': str, 'solution': str, 'level': str, 'type': str}
# Note: answers are in LaTeX \boxed{} format: "...= \boxed{42}"
```

### Converting to nanochat Chat Format

nanochat's SFT expects JSONL with "messages" field:

```python
# mathnano/data/prepare_sft.py

import json

def gsm8k_to_chat(example):
    # Extract the clean answer (after ####)
    raw_answer = example['answer']
    steps, answer = raw_answer.split('####')
    answer = answer.strip()
    
    # Format steps to be cleaner
    solution = steps.strip() + f"\n\nThe answer is **{answer}**."
    
    return {
        "messages": [
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": solution}
        ]
    }

def math_to_chat(example):
    return {
        "messages": [
            {"role": "user", "content": example['problem']},
            {"role": "assistant", "content": example['solution']}
        ]
    }

def openmathinstruct_to_chat(example):
    return {
        "messages": [
            {"role": "user", "content": example['problem']},
            {"role": "assistant", "content": example['generated_solution']}
        ]
    }

# Write combined JSONL (80% OMI, 10% GSM8K, 10% MATH)
with open("mathnano/data/processed/sft_combined.jsonl", "w") as f:
    # Sample from OpenMathInstruct (large — sample 100K to keep it manageable)
    for ex in load_dataset("nvidia/OpenMathInstruct-2", split="train").shuffle().select(range(100000)):
        f.write(json.dumps(openmathinstruct_to_chat(ex)) + "\n")
    
    # All of GSM8K train
    for ex in load_dataset("gsm8k", "main", split="train"):
        f.write(json.dumps(gsm8k_to_chat(ex)) + "\n")
    
    # All of MATH train
    for ex in load_dataset("hendrycks/competition_math", split="train"):
        f.write(json.dumps(math_to_chat(ex)) + "\n")
```

---

## Stage 3: GRPO Datasets

GRPO needs problems with verified final answers. We use the same GSM8K and MATH
datasets, formatted differently — just the problem and the ground-truth answer
(no solution steps).

```python
# mathnano/data/prepare_grpo.py

import json, re

def extract_gsm8k_answer(answer_str):
    # GSM8K final answer is after ####
    return answer_str.split('####')[1].strip()

def extract_math_answer(solution_str):
    # MATH final answer is in \boxed{}
    match = re.search(r'\\boxed\{([^}]+)\}', solution_str)
    return match.group(1) if match else None

with open("mathnano/data/processed/grpo_problems.jsonl", "w") as f:
    for ex in load_dataset("gsm8k", "main", split="train"):
        f.write(json.dumps({
            "problem": ex['question'],
            "answer": extract_gsm8k_answer(ex['answer'])
        }) + "\n")
    
    for ex in load_dataset("hendrycks/competition_math", split="train"):
        answer = extract_math_answer(ex['solution'])
        if answer:  # Skip if answer extraction fails
            f.write(json.dumps({
                "problem": ex['problem'],
                "answer": answer
            }) + "\n")
```

---

## Data Quality Notes

### MathPile cautions

1. **arXiv preambles**: many arXiv papers start with LaTeX preambles (`\documentclass`,
   `\usepackage`, etc.) with no mathematical content. These should be filtered or
   the tokenizer will allocate vocabulary to LaTeX commands that aren't math.
   nanochat's tokenizer training handles this, but be aware.

2. **Bibliography sections**: the last portion of many arXiv papers is just citations.
   These add tokens but no mathematical value. You can truncate documents at a
   `\bibliographystyle` marker if desired.

3. **Language**: most of MathPile is English but some arXiv papers are in other
   languages. This is fine — it won't hurt the model.

### GSM8K cautions

1. Some problems have multiple valid interpretations. The "correct" answer is the
   one in the dataset, but a model that gets a different valid interpretation will
   be counted as wrong. This is a known limitation of the benchmark.

2. Answers are always integers or simple fractions. Good for our extractor.

### MATH cautions

1. The 5 difficulty levels are not equally distributed. Level 5 (hardest) problems
   are rare. Our model will likely score near 0% on Level 4–5.

2. Some solutions use advanced LaTeX that our tokenizer may handle poorly.
   Check a sample of solutions before training.

3. Answer format is `\boxed{answer}`. Make sure your reward extractor handles
   nested braces: `\boxed{\frac{1}{2}}`.

---

## Estimated Storage

| Dataset              | Raw size   | Tokenised size         |
|----------------------|------------|------------------------|
| MathPile             | ~50 GB     | ~18.8 GB (uint16 bin)  |
| OpenMathInstruct-2   | ~4 GB      | ~500 MB (JSONL)        |
| GSM8K                | ~10 MB     | ~5 MB (JSONL)          |
| MATH                 | ~50 MB     | ~20 MB (JSONL)         |
| GRPO problems        | ~5 MB      | ~2 MB (JSONL)          |
| **Total**            | **~55 GB** | **~20 GB**             |

Plan for 60–70 GB of storage on your RunPod pod or persistent volume.

---

## HuggingFace Hub (Backup)

Before closing any pod, push processed data to HuggingFace:
```bash
huggingface-cli upload mathnano/data/processed/ your-username/mathnano-data \
  --repo-type dataset
```

This means you can re-download the processed data on any future pod without
re-running the (slow) processing scripts.
