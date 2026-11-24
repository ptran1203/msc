"""Self-hosted HF model backend for the POC graders (Qwen3 text LLMs and Qwen2.x-VL used text-only).

Loaded lazily and cached, so only the first call pays the load time. Needs torch + transformers,
e.g. the VQA_Med venv: ~/workspace/codebase/VQA_Med/.venv/bin/python
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

MAX_NEW_TOKENS = 128  # the reply is one small JSON object


def _device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load(model_id: str) -> tuple[Any, Any, bool]:
    """Return (model, tokenizer_or_processor, is_vision_model)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    is_vl = hasattr(AutoConfig.from_pretrained(model_id), "vision_config")
    model_cls = AutoModelForImageTextToText if is_vl else AutoModelForCausalLM
    model = model_cls.from_pretrained(model_id, dtype=torch.bfloat16).to(_device(torch)).eval()
    tok = AutoProcessor.from_pretrained(model_id) if is_vl else AutoTokenizer.from_pretrained(model_id)
    return model, tok, is_vl


def generate(prompt: str, model_id: str) -> str:
    """Greedy-decode one chat turn and return the reply text."""
    import torch

    model, tok, is_vl = _load(model_id)
    if is_vl:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        extra = {}
    else:
        messages = [{"role": "user", "content": prompt}]
        extra = {"enable_thinking": False}  # Qwen3: skip <think> to keep replies short and fast
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                     return_dict=True, return_tensors="pt", **extra).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)
