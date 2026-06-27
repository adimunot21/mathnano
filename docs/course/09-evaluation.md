# 9. Evaluation: how to know if it actually works

You cannot improve what you can't measure, and in ML it's terrifyingly easy to fool yourself.
Evaluation is the discipline of getting an honest number for "how good is this model, really." It's
also what told us SFT worked (39/40%) and GRPO failed (5.5%) — without it we'd have been guessing.

## 9.1 Benchmarks: standardized exams for models

A **benchmark** is a fixed set of problems with known answers, held out from training, that everyone
reports on so results are comparable. For math:
- **GSM8K** — 8.5k grade-school word problems (multi-step arithmetic/reasoning). Test set: 1,319.
- **MATH** — 12.5k competition problems across 7 subjects and 5 difficulty levels. Much harder.

▶ **In MathNano** we used GSM8K's test split and a mirror of MATH (the original was taken down;
finding a faithful mirror, `nlile/hendrycks-MATH-benchmark`, was its own "inspect the data" task).
We evaluated on `n=200` per benchmark to keep cost down while still being statistically meaningful.

## 9.2 Accuracy, and the subtlety of "correct"

For verifiable tasks, the metric is **accuracy**: fraction of problems where the model's extracted
answer matches the ground truth. The subtlety is entirely in "matches": is `0.5` = `1/2`? Is
`\boxed{42}` = "the answer is 42"? Is `12 km` = `12`? A naive string comparison says no to all of
these and *understates* your model. Our checker normalizes formatting and tests numeric/symbolic
equivalence so it credits genuinely-correct answers regardless of surface form (Chapter 10).

**The golden rule we followed:** evaluation uses the *exact same* `math_reward` function as RL
training. If eval and training disagree about correctness, your RL is optimizing a different target
than you're reporting — a classic way to ship a "great" model that isn't.

## 9.3 pass@k: measuring potential vs reliability

- **accuracy / pass@1**: one greedy attempt per problem — "is it right the first time?" (reliability)
- **pass@k**: `k` sampled attempts; correct if *any* is right — "can it get there at all?" (potential)

pass@k is always ≥ pass@1 and the gap tells you something: a big gap means the model *can* solve it
but isn't reliable — exactly the situation RL is designed to fix (reward the attempts that land). ▶
Our harness (`mathnano/eval/`) supports both via an `--k` flag.

## 9.4 Per-slice breakdowns

A single accuracy number hides structure. Break it down. ▶ Our MATH eval reports **by difficulty
level**, which is far more informative than the average:

```
L1 66.7%  ·  L2 56.8%  ·  L3 41.5%  ·  L4 34.1%  ·  L5 18.0%
```

That gradient (easy problems mostly solved, hardest mostly missed) is exactly what you'd expect from
a competent small model, and it tells you *where* to invest next. An aggregate "40%" tells you none
of that.

## 9.5 Design for reuse: model-agnostic evaluation

▶ **In MathNano** the eval harness is decoupled from *how* text is generated: it takes any object
with a `.generate(prompts) -> list[str]` method (a `Generator`). That means the *same* harness scores
the from-scratch model, the Qwen SFT model, the GRPO model, or even the live serving API — and it can
be unit-tested with a fake "DummyGenerator" and **zero** ML dependencies (we have tests that verify
the scoring logic in milliseconds). Good evaluation code is reusable infrastructure, not a one-off
script.

## 9.6 Honesty: report what's real

We report `n=200` (not the full test sets), greedy decoding, with a specific answer-checker — and we
say so. We report the GRPO **collapse** alongside the SFT success. Benchmarks also have known flaws
(ambiguous problems, test-set contamination in big models), so a number is evidence, not gospel.
Evaluation done honestly is what separates a credible result from a marketing claim.

## What breaks without this
Without rigorous eval you're flying blind: you can't tell improvement from regression (we'd never
have caught the GRPO collapse), you can't compare models, and you can't make a defensible claim about
your model's quality. Worse, a *sloppy* eval (lenient extraction, train/test leakage, or — the
deadly one — an eval that disagrees with your training reward) actively misleads you into shipping
something broken.

→ Next: [The reward, inference, and serving](10-reward-inference-serving.md)
