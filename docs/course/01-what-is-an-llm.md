# 1. What a language model actually is

Strip away the hype and a large language model is one deceptively small idea:

> A language model is a function that takes a sequence of tokens and returns a probability
> distribution over what the **next** token will be.

That's it. `P(next token | all previous tokens)`. Everything else — attention, RoPE, RLHF, the
£100M training runs — exists to make that single prediction accurate.

## 1.1 Tokens, not words

Models don't see text; they see integers. Text is chopped into **tokens** (roughly: common
word-pieces) and each token is an integer id. "The cat sat" might become `[464, 3797, 3332]`. We
cover *how* the chopping works in [Chapter 2](02-tokenization.md); for now, a token is just an
integer drawn from a fixed **vocabulary** of size `V`.

So more precisely, the model outputs a vector of `V` numbers — one score per possible next token —
which we turn into probabilities with a **softmax** (exponentiate each score, divide by the sum, so
they're all positive and sum to 1).

▶ **In MathNano** our vocabulary is `V = 32,768` tokens. So at every position the model emits 32,768
numbers and says, in effect, "here's how likely each of these 32,768 tokens is to come next."

## 1.2 Why next-token prediction is enough to be useful

This feels too simple to produce something that can solve math. The leap is: **to predict the next
token well across billions of examples, the model is forced to learn structure.** To finish
"The capital of France is ___" it must encode a fact. To finish "2 + 2 = ___" it must encode
arithmetic. To finish "Proof. Suppose for contradiction that ___" it must encode how proofs go.
Compression of text *is* learning, because the cheapest way to predict text is to understand it.

▶ **In MathNano**, after pretraining only on math text, our from-scratch model completes
"The chemical symbol of gold is ___" with "Au", and "If yesterday was Friday, then tomorrow will
be ___" with "Saturday" — knowledge that emerged purely from next-token prediction. It does **not**
reliably solve equations yet; that needs the later stages (Chapters 7–8).

## 1.3 Generation: autoregression

The model predicts *one* token's distribution. To produce a sentence, you do it repeatedly:

1. Feed the prompt, get the distribution for the next token.
2. Pick a token (greedily = the most likely, or by sampling).
3. Append it to the sequence.
4. Repeat from step 1 with the longer sequence.

This loop is **autoregressive generation** — each output becomes part of the next input. It's why
generation is sequential and why long outputs are slow (more on speeding this up — the *KV cache* —
in [Chapter 10](10-reward-inference-serving.md)).

**Temperature** controls step 2's randomness. Temperature 0 = always take the most likely token
(deterministic, "greedy"). Higher temperature flattens the distribution → more diverse, more
creative, more error-prone. ▶ In MathNano we eval at temperature 0 (we want the model's best single
answer) but generate at temperature 1.0 during RL (Chapter 8) because we *want* varied attempts.

## 1.4 What "training" means here

The model is a giant function with millions–billions of tunable numbers (**parameters** / weights).
Training = adjusting those numbers so the predicted distribution puts high probability on the token
that *actually* came next in the training text. The mismatch between "what the model predicted" and
"what actually came next" is the **loss**; training minimizes it via gradient descent
([Chapter 5](05-training.md)).

At the very start, the weights are random, so the model thinks every token is equally likely:
probability `1/V` each. The loss of that uniform guess is `ln(V)` (natural log of the vocab size).

▶ **In MathNano** `ln(32768) ≈ 10.40`. When we started our real pretraining run, step 0 printed
`loss: 10.396975` — within 0.003 of the theoretical maximum. That single number confirmed the whole
stack (tokenizer, model init, data) was wired correctly before we spent a penny on the full run.
Watching that loss fall from 10.4 → 1.44 over 11.8 hours *is* the model learning.

## 1.5 The three things that make a good LLM

1. **Scale** — more parameters and more training data, in a balanced ratio (the *scaling laws*,
   Chapter 12). Frontier models are big; ours is deliberately tiny (~200M and 1.5B).
2. **Architecture** — the transformer (Chapters 3–4) is the design that made scale trainable.
3. **The training recipe** — pretrain → fine-tune → (optionally) RL (Part 2). This is where a raw
   text-predictor becomes a helpful assistant.

The rest of Part 1 builds the machine. Then Part 2 runs the recipe.

## What breaks without this idea
If you don't internalize "it's just next-token prediction," everything else looks like magic. Once
you do, every capability (translation, coding, math, chat) is the same mechanism conditioned on
different context — and every limitation (hallucination, getting confidently wrong answers) is "the
most *probable* continuation isn't always the *true* one." Hold onto that; it explains the whole
field.

→ Next: [Tokenization](02-tokenization.md)
