# 7. Supervised fine-tuning: from autocomplete to assistant

A pretrained model autocompletes text. It has no concept of "question" and "answer" — it treats them
symmetrically, as more text to continue. **Supervised fine-tuning (SFT)** teaches it a specific
behavior: *given a question in a particular format, produce a helpful, well-structured answer.* This
is the step that turns a text predictor into a chatbot.

## 7.1 The idea: same loss, curated data, masked prompt

SFT is just more training (same cross-entropy loss, Chapter 5) but on a curated dataset of
**conversations** rather than raw text: `(system, user question, assistant answer)` triples, encoded
with the chat template's special tokens (Chapter 2).

One crucial twist: we only want the model to learn to produce the **assistant's** tokens, not to
predict the user's question. So ideally we **mask** the prompt tokens out of the loss (compute loss
only on the answer). "Full-sequence" SFT (loss on everything) also works and is simpler; it just
spends some capacity modeling questions too.

▶ **In MathNano** we used full-sequence SFT — because the exact TRL library version we had
(`trl==0.16`) didn't expose the `assistant_only_loss` option (it arrived in a later release). A small
compromise, noted honestly in the code; the results were strong regardless. (This is the kind of
version-reality friction Chapter 11 is about.)

## 7.2 Our setup: LoRA on Qwen2.5-1.5B

Two tracks ran the recipe. **Track A** kept fine-tuning our from-scratch 200M model. **Track B** —
the one we actually shipped — fine-tuned a small *pretrained open* base, **Qwen2.5-1.5B**, because
starting from a model that already speaks fluent English/math gives a far more capable product than
200M-from-scratch can, for the same effort. (Choosing the right base is half the battle; see
Chapter 12.)

### LoRA: cheap fine-tuning
Full fine-tuning updates all 1.5B weights — memory-heavy. **LoRA (Low-Rank Adaptation)** freezes the
base model and inserts tiny trainable "adapter" matrices (rank 16 here) into each layer. You train
~1% of the parameters, get ~full-fine-tune quality on these tasks, and the result is a **74 MB
adapter** you load on top of the base. It fit comfortably on a 24 GB GPU and is trivial to share
(it's on HuggingFace as `adimunot/mathnano-qwen1.5b-sft`).

## 7.3 The data, and a design choice that mattered

We built ~119k SFT examples from GSM8K (grade-school word problems), the MATH competition dataset,
and a 100k subset of OpenMathInstruct-2 (high-quality chain-of-thought solutions). Every assistant
answer was reformatted to end with a standardized line:
```
The final answer is $\boxed{<answer>}$.
```
**Why force the `\boxed{}` format?** Because our reward and evaluation (Chapters 8–9) extract the
answer by looking for `\boxed{}`. Training the model to *always* box its final answer means the
answer is reliably machine-checkable — which makes both evaluation and RL clean. Design the model's
output format around how you'll *measure* it.

## 7.4 The result

SFT was a clear success:
- training loss fell to ~0.48, token accuracy ~87%;
- on held-out tests: **GSM8K 39.0%, MATH 40.0%** (n=200 each), with MATH ranging from 67% on the
  easiest level to 18% on the hardest.

For a 1.5B model fine-tuned on a £25-budget project, 39–40% is a genuinely respectable result, and
it's the model we serve. The contrast with the pretrained-only model is stark: the same "5x − 3 = 12"
style question it used to get wrong, it now solves step by step and boxes the right answer.

## 7.5 SFT vs pretraining, in one line
Pretraining taught the model *the language of math*; SFT taught it *the behavior of solving and
presenting* — given a problem, produce a clean worked solution ending in a boxed answer. It didn't
add much new knowledge; it shaped the knowledge into a usable form.

## What breaks without this
Without SFT you have a model that rambles plausible math but won't reliably answer your actual
question in a usable format — it autocompletes instead of assisting. SFT is the cheapest, highest-
leverage step for turning raw capability into a product. (The remaining gap — being *correct* more
often, not just well-formatted — is what RL targets next.)

→ Next: [Reinforcement learning with verifiable rewards](08-rl-grpo.md)
