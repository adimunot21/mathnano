"""Generation backends for the eval harness and the product.

`HFGenerator` wraps a HuggingFace causal LM (Track B / Qwen, with optional LoRA adapter). It
builds prompts with the tokenizer's chat template — the SAME template SFT used — so eval matches
deployment. Heavy imports (torch/transformers) are lazy so the harness imports cleanly in a
minimal env; only constructing an HFGenerator needs the ML stack.

A nanochat backend (Track A) can be added later by wrapping `nanochat.engine.Engine` behind the
same `.generate()` signature.
"""
from __future__ import annotations

from typing import Optional, Sequence

from mathnano.eval.runner import DEFAULT_SYSTEM


class HFGenerator:
    def __init__(self, model_id: str, *, adapter: Optional[str] = None, device: str = "auto",
                 dtype: str = "auto", load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # Stop generation at the chat turn-end for Qwen-style chat models, so completions end
        # after the answer instead of running to max_new_tokens (faster + clean extraction).
        if "<|im_end|>" in self.tok.get_vocab():
            self.tok.eos_token = "<|im_end|>"
        torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[dtype]
        kwargs = dict(torch_dtype=torch_dtype, device_map=device)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, prompts: Sequence[str], *, system: str = DEFAULT_SYSTEM,
                 temperature: float = 0.0, max_new_tokens: int = 512) -> list[str]:
        import torch

        texts = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       padding_side="left").to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                          eos_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        # strip the prompt tokens; decode only the newly generated continuation
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(new, skip_special_tokens=True)
