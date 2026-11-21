"""Local open-weight model client (4-bit quantized) for the prompting baseline."""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
SEED = 42


class LocalLLM:
    def __init__(self, model_name: str = MODEL_NAME, seed: int = SEED, adapter: str | None = None):
        torch.manual_seed(seed)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # T4 (Kaggle) has no native bf16 -- fp16 there, bf16 on Ampere+ (local 3060)
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, system: str, user: str, max_new_tokens: int = 512, temperature: float | None = None, seed: int | None = None) -> str:
        """temperature=None -> deterministic greedy (the default, reproducible on its own).
        temperature set -> sampling, seeded explicitly so it's still reproducible; used for
        a retry attempt when greedy decoding deterministically repeats a failure mode."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if temperature is not None:
            torch.manual_seed(seed if seed is not None else SEED)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature is not None,
                temperature=temperature,
                top_p=0.9 if temperature is not None else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_code_block(text: str) -> str | None:
    """Pull a fenced ```cpp ... ``` (or bare ```) code block out of a model response."""
    m = re.search(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_json(text: str) -> dict | None:
    """Pull the first well-formed JSON object out of a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None
