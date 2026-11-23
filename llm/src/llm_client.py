"""Local open-weight model client (4-bit quantized) for the prompting baseline."""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
SEED = 42


class LocalLLM:
    def __init__(self, model_name: str = MODEL_NAME, seed: int = SEED, adapter: str | None = None,
                 thinking: bool = False, think_budget: int = 1024):
        """thinking=True: Qwen3-style hybrid models think in <think>...</think> before answering
        (chat template's enable_thinking). thinking is capped at think_budget tokens (budget forcing, see generate), and
        the think block is stripped from the returned text so callers parse only the answer."""
        torch.manual_seed(seed)
        self.model_name = model_name
        self.thinking = thinking
        self.think_budget = think_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # T4 (Kaggle) has no native bf16 -- fp16 there, bf16 on Ampere+ (local 3060)
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            # unquantized parts (embeddings, lm_head) default to fp32 otherwise: Qwen3-8B's
            # untied 151k-vocab embed+head alone is ~5GB fp32 -> 12GB total, WDDM spill on the 3060
            torch_dtype=compute_dtype,
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
        # explicit both ways: Qwen3's template thinks by default; other templates ignore the kwarg
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.thinking)
        if self.thinking and temperature is None:  # Qwen3 docs: greedy decoding loops in thinking mode -> seeded sampling
            temperature = 0.6
        if temperature is not None:
            torch.manual_seed(seed if seed is not None else SEED)
        if not self.thinking:
            return self._generate(text, max_new_tokens, temperature)
        # Budget forcing: think for at most think_budget tokens. Unfinished thinking (Qwen3-8B
        # rambled >3000 tokens on a multi-problem item) is closed with Qwen's own early-stop
        # phrase, then the answer gets its full max_new_tokens.
        gen, finished = self._generate(text, self.think_budget, temperature, return_finished=True)
        if not finished:
            if "</think>" not in gen:
                gen += "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
            gen += self._generate(text + gen, max_new_tokens, temperature)
        return gen.rsplit("</think>", 1)[1] if "</think>" in gen else gen

    def _generate(self, text: str, max_new_tokens: int, temperature: float | None, return_finished: bool = False):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature is not None,
                temperature=temperature,
                top_p=(0.95 if self.thinking else 0.9) if temperature is not None else None,
                pad_token_id=self.tokenizer.eos_token_id,
                **({"top_k": 20} if self.thinking else {}),  # Qwen3 thinking-mode recommendation
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if self.thinking:  # long, varying-length generations grew the cache to 12.1GB (WDDM cliff)
            torch.cuda.empty_cache()
        if not return_finished:
            return decoded
        # finished = stopped on an end token before the budget ran out
        return decoded, len(new_tokens) < max_new_tokens


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
