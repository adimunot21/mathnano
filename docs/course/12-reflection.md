# 12. Scaling laws, what we achieved, and where small models win

You've built the machine (Part 1), run the recipe (Part 2), and shipped it (Part 3). This closing
chapter zooms out: why size matters so much, what we honestly did and didn't achieve, and where a
small model you own can still beat a giant you rent.

## 12.1 Scaling laws: why bigger really is better

One of the most robust empirical findings in ML: model quality improves *predictably* with scale.
Loss falls as a smooth power law in three things — **parameters**, **training data**, and
**compute** — over many orders of magnitude. Two consequences shaped this project:

- **Compute-optimal balance (Chinchilla).** For a given compute budget, there's an optimal ratio of
  model size to training tokens (~20 tokens per parameter, give or take). Too few tokens for your
  size and you're wasting parameters; too many and you're wasting compute. ▶ This is why we trained
  our ~200M model on ~2.5B tokens, not all 9.5B of MathPile — and why over-training (epoch 7) started
  to overfit (Chapter 6). nanochat literally derives the training length from this rule.
- **You cannot small-model your way to frontier quality.** A 1.5B model is ~1000× smaller than a
  frontier model. No clever trick closes a 1000× gap — the scaling curve is the scaling curve. This
  is the honest ceiling on what £13 buys.

## 12.2 What we actually achieved (and didn't)

**Didn't:** build something competitive with ChatGPT. Our solver gets ~39% on GSM8K; frontier models
get ~95%. As a *product competing with ChatGPT*, it loses, and was never going to win at this scale.
If that was the goal, the goal was impossible.

**Did:** the thing that's actually valuable —
- Trained a transformer **from random initialization** into a coherent math language model (bpb
  0.731), understanding every component and every number along the way.
- Ran the **full frontier recipe** — pretrain → SFT → RL-with-verifiable-rewards (the DeepSeek-R1
  recipe) — on a single consumer GPU for ~£13, and can explain each stage.
- Built a **measurably better** model via fine-tuning (39/40% on GSM8K/MATH), proven with an
  evaluation harness we wrote.
- Produced an **honest, well-diagnosed negative result** (the GRPO collapse and its reward-fidelity
  root cause) — often a stronger demonstration of understanding than a success.
- Shipped it: a tested reward, a reusable eval harness, an inference API + UI, reproducible infra.

The distinction that matters: **ChatGPT is a thing you use; this is a thing you can build.** Using a
model and building/training/evaluating/debugging one are different skills, and the second is what
this whole course is about.

## 12.3 Where a small model you own actually wins

"Worse than ChatGPT in general" is not "useless." A small, owned model beats a giant API when the
axis isn't raw capability:

- **Cost & latency at volume.** No per-call fees, no rate limits. If you run millions of math checks,
  a £13 model on your own hardware is far cheaper than an API.
- **Privacy & control.** It runs entirely offline. Sensitive data never leaves your machine — often a
  hard requirement (healthcare, finance, on-device).
- **Specialization.** This is the real edge. A small model **fine-tuned on data the giant has never
  seen** — your company's internal problems, a niche notation, a proprietary domain — can *beat* a
  general frontier model *on that specific task*. You can't fine-tune ChatGPT on your private corpus;
  you can fine-tune this.
- **Ownership.** It can't be deprecated, rate-limited, price-hiked, or have its behavior changed out
  from under you. It's yours.

## 12.4 If you want to take it further

In rough order of impact:
1. **Fix and rerun GRPO** with clean rollouts + a robust reward (Chapter 8) — close the loop and
   show RL *improving* over SFT, the recipe's whole point.
2. **Specialize to a domain you have private data for** — the one game a small model can genuinely
   win (12.3). This is how "a learning project" becomes "a useful tool."
3. **Scale within budget** — a larger base (3B–7B) with LoRA, or more SFT data, moves the numbers.
4. **Serve it properly** — GPU inference, batching, a public HuggingFace Space for a portfolio demo.
5. **Add capabilities** — tool use (let it call a calculator/Python — huge for arithmetic
   reliability), longer context, more benchmarks.

## 12.5 The real takeaway

Every capability of every LLM, from this 200M math model to GPT-4, is the same idea: **next-token
prediction**, made accurate by **scale**, structured by the **transformer**, and shaped into
usefulness by **pretrain → fine-tune → RL**. The labs have more compute and more data; they do not
have a different secret. You now understand the entire stack — not as magic, but as a sequence of
comprehensible, debuggable engineering decisions you've made yourself.

That understanding — not the 39% — is what you actually built.

← Back to the [course index](README.md).
