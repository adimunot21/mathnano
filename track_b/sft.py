r"""Track B — Supervised fine-tuning of Qwen2.5-1.5B on our math SFT data (LoRA).

Teaches the base model our solve-then-\boxed format on GSM8K + MATH + OpenMathInstruct-2
(the chat JSONL from `mathnano.data.prepare_sft`). LoRA keeps it comfortably within a 24 GB 4090
and produces a small adapter we serve / further GRPO.

⚠️ Run a 5-step smoke FIRST (``--max-steps 5``) to confirm the TRL/transformers versions on the
pod accept this config, THEN the full run. TRL's API drifts between versions — if a kwarg is
rejected, check `SFTConfig`'s signature for that version and adjust.

Usage (in the Track B venv):
    python track_b/sft.py --max-steps 5                       # smoke
    python track_b/sft.py                                     # full run
    python track_b/sft.py --data mathnano/data/processed/sft_combined.jsonl \
        --base Qwen/Qwen2.5-1.5B --out track_b/outputs/sft
"""
from __future__ import annotations

import argparse

CONFIG = {
    "base": "Qwen/Qwen2.5-1.5B",                 # general base (chosen for the biggest SFT->GRPO jump)
    "data": "mathnano/data/processed/sft_combined.jsonl",
    "out": "track_b/outputs/sft",
    "max_seq_len": 1024,                          # covers our problem+CoT; keeps memory modest
    "epochs": 2,
    "lr": 2e-4,                                   # typical LoRA SFT LR
    "per_device_batch": 8,
    "grad_accum": 4,                             # effective batch 32
    "warmup_ratio": 0.03,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "seed": 1337,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Track B SFT (LoRA) on Qwen2.5-1.5B")
    ap.add_argument("--base", default=CONFIG["base"])
    ap.add_argument("--data", default=CONFIG["data"])
    ap.add_argument("--out", default=CONFIG["out"])
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke test)")
    ap.add_argument("--epochs", type=float, default=CONFIG["epochs"])
    args = ap.parse_args()

    # Heavy imports inside main so `python track_b/sft.py --help` is instant.
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Conversational dataset: TRL applies the Qwen chat template to the `messages` column.
    ds = load_dataset("json", data_files=args.data, split="train")

    lora = LoraConfig(
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )

    cfg = SFTConfig(
        output_dir=args.out,
        max_length=CONFIG["max_seq_len"],
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=CONFIG["per_device_batch"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        learning_rate=CONFIG["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=CONFIG["warmup_ratio"],
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        # NOTE: full-sequence SFT (TRL 0.16's SFTConfig has no assistant_only_loss). For our
        # focused solver this is fine and GRPO does the heavy lifting on correctness; if SFT eval
        # underperforms we can add completion-only masking via DataCollatorForCompletionOnlyLM.
        report_to="none",
        seed=CONFIG["seed"],
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )

    trainer = SFTTrainer(
        model=args.base,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tok,
    )
    print(f"[sft] base={args.base}  examples={len(ds):,}  out={args.out}")
    trainer.train()
    trainer.save_model(args.out)          # saves the LoRA adapter
    tok.save_pretrained(args.out)
    print(f"[sft] done -> adapter at {args.out}")


if __name__ == "__main__":
    main()
