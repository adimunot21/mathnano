r"""Track B — GRPO with verifiable rewards on Qwen2.5-1.5B (the headline result).

Starts from the SFT adapter and improves *correctness* (not just format) via GRPO: for each
problem the model samples G solutions, our shared verifiable reward scores them (+1 correct /
-1 wrong), and the policy is pushed toward the higher-reward samples. Reward =
`mathnano.rewards.math_reward.math_reward` — the EXACT function the eval harness uses, so training
and evaluation never disagree about "correct".

⚠️ Smoke FIRST (``--max-steps 5``). The single-GPU TRL+vLLM combo is the fiddly part:
  * `num_generations` must divide the effective batch; if TRL complains, adjust
    `--per-device-batch` / `--num-generations` per the error.
  * vLLM and the trainer share the 24 GB GPU — tune `--vllm-mem` (vLLM's fraction) down if OOM,
    or pass `--no-vllm` to use slower HF generation.

Usage (in the Track B venv, from repo root so `mathnano` imports):
    python track_b/grpo.py --sft-adapter track_b/outputs/sft --max-steps 5      # smoke
    python track_b/grpo.py --sft-adapter track_b/outputs/sft                    # full run
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `mathnano` importable when run as `python track_b/grpo.py` (script dir, not repo root,
# is what Python puts on sys.path by default).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathnano.rewards.math_reward import math_reward  # shared verifiable reward

CONFIG = {
    "base": "Qwen/Qwen2.5-1.5B",
    "data": "mathnano/data/processed/grpo_problems.jsonl",
    "out": "track_b/outputs/grpo",
    "system": ("You are a careful mathematician. Solve the problem step by step, "
               "then give the final answer in \\boxed{}."),
    "num_generations": 8,            # G: samples per problem
    "per_device_batch": 16,          # must be divisible by num_generations (see note above)
    "grad_accum": 4,
    "lr": 1e-6,                      # small LR for RL stability
    "beta": 0.04,                   # KL penalty vs the reference (SFT) model — prevents drift/hacking
    "temperature": 1.0,             # high temp = exploration across solution strategies
    "max_prompt_len": 512,
    "max_completion_len": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "seed": 1337,
}


def reward_correct(completions, answer, **kwargs):
    """TRL reward fn: +1 if the sampled solution's answer matches ground truth, else -1.

    `completions` is conversational (list of message-lists) or plain strings depending on the TRL
    version / dataset; we extract the text defensively. `answer` is the per-example column.
    """
    texts = []
    for c in completions:
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list) and c and isinstance(c[-1], dict):
            texts.append(c[-1].get("content", ""))   # last assistant turn
        else:
            texts.append(str(c))
    return [math_reward(t, a) for t, a in zip(texts, answer)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Track B GRPO on Qwen2.5-1.5B")
    ap.add_argument("--base", default=CONFIG["base"])
    ap.add_argument("--sft-adapter", default=None, help="SFT LoRA adapter to start from (merged in)")
    ap.add_argument("--data", default=CONFIG["data"])
    ap.add_argument("--out", default=CONFIG["out"])
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--num-generations", type=int, default=CONFIG["num_generations"])
    ap.add_argument("--per-device-batch", type=int, default=CONFIG["per_device_batch"])
    ap.add_argument("--max-completion-len", type=int, default=CONFIG["max_completion_len"],
                    help="hard cap on generated tokens (backstop if the model doesn't emit eos)")
    ap.add_argument("--no-vllm", action="store_true", help="use HF generation instead of vLLM")
    ap.add_argument("--vllm-mem", type=float, default=0.3, help="vLLM GPU memory fraction")
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # TRL drives generation off the TOKENIZER's eos. Qwen's default eos is <|endoftext|>, which the
    # SFT model rarely emits — so rollouts ran to max length. Point eos at the chat turn-end so
    # completions stop right after the answer (faster + clean extraction). pad stays distinct.
    if "<|im_end|>" in tok.get_vocab():
        tok.eos_token = "<|im_end|>"

    # Build the dataset: conversational prompt + the ground-truth answer column for the reward.
    raw = load_dataset("json", data_files=args.data, split="train")

    def to_prompt(ex):
        return {"prompt": [{"role": "system", "content": CONFIG["system"]},
                           {"role": "user", "content": ex["problem"]}],
                "answer": ex["answer"]}

    ds = raw.map(to_prompt, remove_columns=[c for c in raw.column_names if c not in ("answer",)])

    # Start policy = base (+ merged SFT adapter, so GRPO continues from the instruction-tuned model).
    model = args.base
    if args.sft_adapter:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="bfloat16")
        merged = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload()
        model = merged
        print(f"[grpo] merged SFT adapter {args.sft_adapter} into base")

    # Stop rollouts at Qwen's chat turn-end (<|im_end|>) so completions end after the answer
    # instead of running to max_completion_length. WHY: the model's default eos is <|endoftext|>,
    # which the SFT model rarely emits — without this, every generation hits 512 tokens (slow,
    # and the answer extractor can grab rambled text after the real answer).
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and not isinstance(model, str):
        stops = sorted({tok.eos_token_id, im_end_id} - {None})
        model.generation_config.eos_token_id = stops
        model.config.eos_token_id = stops
        print(f"[grpo] generation stop tokens: {stops}")

    lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
                      target_modules="all-linear", bias="none", task_type="CAUSAL_LM")

    cfg = GRPOConfig(
        output_dir=args.out,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=CONFIG["grad_accum"],
        learning_rate=CONFIG["lr"],
        beta=CONFIG["beta"],
        temperature=CONFIG["temperature"],
        max_prompt_length=CONFIG["max_prompt_len"],
        max_completion_length=args.max_completion_len,
        max_steps=args.max_steps,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=not args.no_vllm,
        vllm_gpu_memory_utilization=args.vllm_mem,
        report_to="none",
        seed=CONFIG["seed"],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_correct,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tok,
    )
    print(f"[grpo] problems={len(ds):,}  G={args.num_generations}  vllm={cfg.use_vllm}  out={args.out}")
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[grpo] done -> adapter at {args.out}")


if __name__ == "__main__":
    main()
