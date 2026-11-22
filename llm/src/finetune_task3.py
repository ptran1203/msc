"""QLoRA fine-tune of Qwen2.5-Coder-7B for Task 3 (self-diagnose + feedback), trained on
the synthetic bug-injection data, evaluated on the 32 gold-labeled sample_dataset items.

Synth data has no Task 3 labels (no oracle for feedback text), so step 1 creates them,
GROUNDED on facts known by construction (injected bug type, exact diff vs the reference
solution, oracle rubric, real g++ log). Whatever can be decided deterministically is:
Level 1 is templated and compile-error Level 2 quotes the g++ error verbatim. A stronger
teacher model only writes the reasoning + the rest of Level 2/3 -- a 7B teacher got about
half of these wrong in a smoke test (e.g. inverted io_format fixes, false edge-case facts).

    python finetune_task3.py gen-targets [--num-shards 2 --shard-id 0]
    python finetune_task3.py train
    python finetune_task3.py eval [--base]
"""
import argparse
import difflib
import json
import math
import random
import re

import torch

import prompts as P
from compile_check import attach_real_compile_log, first_compile_error
from data import REPO_ROOT, load_dataset, load_sample_dataset, read_code
from llm_client import MODEL_NAME, SEED, LocalLLM, extract_json
from research_task3_ideas import build_self_diagnose_prompt, sanitize, score
from synth_bugs import BUG_TAXONOMY
from synth_oracle import REF_DIR

SYNTH_DIR = REPO_ROOT / "synth/generated"
ADAPTER_DIR = REPO_ROOT / "outputs/ft_task3_lora"
EVAL_DIR = REPO_ROOT / "outputs/ft_task3_eval"
TEACHER_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
LEVELS = ["Level 1 - Gợi ý nhẹ", "Level 2 - Chẩn đoán", "Level 3 - Gợi ý sửa"]
EXAM_TO_PROBLEM = {
    "SYN-ARMSTRONG": "armstrong", "SYN-DEDUP": "remove_duplicates", "SYN-LOGIN": "earliest_login",
    "SYN-BINSEARCH": "binary_search", "SYN-PALINDROME": "palindrome_check",
}
LEVEL1_BUGGY = ["Chưa đúng, em xem lại.", "Code chưa đúng, em kiểm tra lại.", "Bài chưa đạt, em xem lại nhé."]
# style_error code runs correctly by construction (oracle-verified), but the 7B teacher
# invented logic bugs for it -- so its Level 2/3 are templated too, never LLM-written.
STYLE_L2 = "Code chạy đúng nhưng tên biến khó hiểu, trình bày rối."
STYLE_L3 = "Code chạy đúng; em đặt tên biến rõ nghĩa và tách mỗi lệnh một dòng cho dễ đọc."
MAX_TRAIN_TOKENS = 2048

# Style rules distilled from the 32 gold feedbacks, restated as patterns rather than
# quoted gold text, so the eval set doesn't leak straight into training targets.
TARGET_SYSTEM = "Bạn là giảng viên lập trình C++ người Việt, viết nhận xét ngắn gọn cho sinh viên."
STYLE_RULES = """Phong cách nhận xét của giảng viên thật (BẮT BUỘC theo):
- Rất ngắn: thường 1 câu, tối đa ~25 từ. Xưng "em". Không chào hỏi, không giải thích dài.
- Level 2 (Chẩn đoán): nói lỗi ở đâu và vì sao, KHÔNG nói cách sửa (dạng "<chỗ sai> bị sai vì <nguyên nhân>.").
- Level 3 (Gợi ý sửa): chẩn đoán + hướng sửa ngắn, có thể nêu số dòng (dạng "<chỗ sai> sai, em <hướng sửa> ở dòng <n>.").
- Nếu là lỗi định dạng output: nói rõ output của em sai thế nào so với yêu cầu đề bài.
- Nếu là lỗi style (code chạy đúng): Level 2/3 nhận xét về tên biến / cách trình bày, không nói code sai.
- Các dạng trong ngoặc chỉ là KHUÔN, phải thay bằng nội dung của ĐÚNG lỗi đã biết ở trên -- KHÔNG chép lại khuôn.
- Chỉ viết tiếng Việt."""


def parse_bug_id(sample) -> str:
    prefix = sample["sample_id"].rsplit("-", 1)[0]
    return prefix[len(sample["input"]["exam_id"]) + 1:]


def load_synth_task3():
    """Synth ships Task 1/2 only -- build Task 3-shaped samples from them (input +
    taxonomy_error + rubric), with the real g++ log attached."""
    ds = load_dataset(SYNTH_DIR)
    rubric_by_id = {s["sample_id"]: s["output"]["rubric"] for s in ds.tasks[1]}
    ds.tasks[3] = []
    for s in ds.tasks[2]:
        inp = dict(s["input"], taxonomy_error=s["output"]["taxonomy_error"])
        ds.tasks[3].append({"sample_id": s["sample_id"], "input": inp, "rubric": rubric_by_id[s["sample_id"]]})
    attach_real_compile_log(ds, 3)
    return ds


def target_prompt(ds, sample) -> str:
    code = read_code(ds, sample)
    problem = EXAM_TO_PROBLEM[sample["input"]["exam_id"]]
    ref = (REF_DIR / f"{problem}.cpp").read_text(encoding="utf-8")
    diff = "\n".join(difflib.unified_diff(ref.splitlines(), code.splitlines(), "dung.cpp", "sinhvien.cpp", n=1, lineterm=""))
    bug = BUG_TAXONOMY[parse_bug_id(sample)]
    numbered = "\n".join(f"{i + 1:3d} | {line}" for i, line in enumerate(code.splitlines()))
    return f"""{P._problem_context(ds, sample)}

## Code sinh viên (có số dòng)
```cpp
{numbered}
```

## SỰ THẬT ĐÃ BIẾT (chỉ để bạn viết đúng -- KHÔNG được nhắc tới "diff", "lời giải mẫu", "bản đúng" trong câu trả lời)
- Loại lỗi: {bug["label"]} ({bug["instruction"]})
- Khác biệt so với lời giải đúng. Dòng bắt đầu bằng "-" là code ĐÚNG, dòng "+" là code của SINH VIÊN (đang sai).
  Nhận xét phải hướng sinh viên sửa từ dòng "+" về dòng "-", KHÔNG được làm ngược lại:
```diff
{diff}
```
- Kết quả chấm tự động (rubric): {json.dumps(sample["rubric"], ensure_ascii=False)}
- compile_log: {sample["input"]["compile_log"]}

{STYLE_RULES}

Viết "reasoning" như thể bạn TỰ đọc code mà phát hiện ra lỗi (2-3 câu, nêu dòng lỗi), rồi viết nhận xét Level 2 và Level 3.
Trả lời CHỈ một JSON object:
{{"reasoning": "...", "level2": "...", "level3": "..."}}
"""


def adapter_dir(model: str):
    # 7B keeps the plain path the Kaggle notebook expects; other students get their own
    if model == MODEL_NAME:
        return ADAPTER_DIR
    return REPO_ROOT / f"outputs/ft_task3_lora_{model.split('/')[-1].lower()}"


def stratified(samples, per_bug: int):
    """First `per_bug` samples of every (exam, bug type) cell -- a small subset that
    still covers all 5 problems x 10 bug types, for quick local checks."""
    counts, picked = {}, []
    for s in samples:
        key = (s["input"]["exam_id"], parse_bug_id(s))
        if counts.get(key, 0) < per_bug:
            counts[key] = counts.get(key, 0) + 1
            picked.append(s)
    return picked


def gen_targets(teacher: str, num_shards: int, shard_id: int, limit=None, per_bug=None):
    ds = load_synth_task3()
    samples = stratified(ds.tasks[3], per_bug) if per_bug else ds.tasks[3]
    samples = samples[:limit] if limit else samples
    samples = samples[shard_id::num_shards]
    out_path = SYNTH_DIR / f"task3_ft_targets_shard{shard_id}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["sample_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    llm = LocalLLM(model_name=teacher)
    n_fail = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if s["sample_id"] in done:
                continue
            bug_id = parse_bug_id(s)
            parsed = extract_json(llm.generate(TARGET_SYSTEM, target_prompt(ds, s), max_new_tokens=400))
            if not parsed or not all(isinstance(parsed.get(k), str) and parsed[k].strip() for k in ("reasoning", "level2", "level3")):
                n_fail += 1
                print(f"[{i + 1}/{len(samples)}] SKIP {s['sample_id']} (unparseable)", flush=True)
                continue
            level1 = "Tốt." if bug_id == "style_error" else random.Random(s["sample_id"]).choice(LEVEL1_BUGGY)
            level2 = first_compile_error(s["input"]["compile_log"]) if bug_id == "compile_error" else parsed["level2"]
            level3 = parsed["level3"]
            if bug_id == "style_error":
                level2, level3 = STYLE_L2, STYLE_L3
            row = {"sample_id": s["sample_id"], "reasoning": parsed["reasoning"], "level1": level1, "level2": level2, "level3": level3}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(samples)}] {s['sample_id']} | L2: {level2[:70]} | L3: {parsed['level3'][:70]}", flush=True)
    print(f"shard {shard_id} done, {n_fail} unparseable skipped -> {out_path}")


def build_train_examples():
    ds = load_synth_task3()
    by_id = {s["sample_id"]: s for s in ds.tasks[3]}
    examples = []
    for path in sorted(SYNTH_DIR.glob("task3_ft_targets_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            s = by_id[t["sample_id"]]
            for level, key in zip(LEVELS, ("level1", "level2", "level3")):
                # Same prompt the model sees at eval/inference: self-diagnose branch, no
                # exemplars, taxonomy_error hidden (build_self_diagnose_prompt never reads it).
                sample = {"sample_id": s["sample_id"], "input": dict(s["input"], target_feedback_level=level)}
                prompt = build_self_diagnose_prompt(ds, sample)
                target = json.dumps({"reasoning": t["reasoning"], "taxonomy_error": s["input"]["taxonomy_error"], "feedback": t[key]}, ensure_ascii=False)
                examples.append((prompt, target))
    return examples


def train(model_name: str, epochs: float, lr: float, max_steps: int):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    use_bf16 = torch.cuda.is_bf16_supported()  # False on Kaggle T4 -> fp16
    out_dir = adapter_dir(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    rows, n_too_long = [], 0
    for prompt, target in build_train_examples():
        msgs = [{"role": "system", "content": P.SYSTEM_GRADER}, {"role": "user", "content": prompt}]
        # render to text then tokenize: apply_chat_template(tokenize=True) returns a list on
        # transformers 4.x but a BatchEncoding on newer versions (Kaggle)
        prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(target_ids) > MAX_TRAIN_TOKENS:
            n_too_long += 1
            continue
        # loss only on the answer, not on re-predicting the prompt
        rows.append({"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids})
    random.Random(SEED).shuffle(rows)
    print(f"{len(rows)} training examples ({n_too_long} dropped > {MAX_TRAIN_TOKENS} tokens), bf16={use_bf16}")

    def collate(batch):
        width = max(len(r["input_ids"]) for r in batch)
        pad = lambda xs, v: xs + [v] * (width - len(xs))
        return {
            "input_ids": torch.tensor([pad(r["input_ids"], tok.pad_token_id) for r in batch]),
            "labels": torch.tensor([pad(r["labels"], -100) for r in batch]),
            "attention_mask": torch.tensor([pad([1] * len(r["input_ids"]), 0) for r in batch]),
        }

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": 0}, torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    grad_accum = 8
    total_steps = max_steps if max_steps > 0 else math.ceil(len(rows) / grad_accum * epochs)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=1, gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs, max_steps=max_steps, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)),  # warmup_ratio was removed in transformers v5
        bf16=use_bf16, fp16=not use_bf16, logging_steps=5, save_strategy="no", report_to=[], seed=SEED,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    Trainer(model=model, args=args, train_dataset=rows, data_collator=collate).train()
    model.save_pretrained(out_dir)
    print(f"saved adapter -> {out_dir}")


def evaluate(model_name: str, base: bool):
    ds = load_sample_dataset()
    attach_real_compile_log(ds, 3)
    valid_labels = set(ds.label_space["taxonomy_error"])
    llm = LocalLLM(model_name=model_name, adapter=None if base else str(adapter_dir(model_name)))
    name = "base_zeroshot" if base else "finetuned"
    if model_name != MODEL_NAME:
        name += "_" + model_name.split("/")[-1].lower()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, s in enumerate(ds.tasks[3]):
        raw = llm.generate(P.SYSTEM_GRADER, build_self_diagnose_prompt(ds, s), max_new_tokens=700)
        parsed = sanitize(extract_json(raw), valid_labels)
        records.append({"sample_id": s["sample_id"], "pred_labels": parsed["taxonomy_error"], "pred_feedback": parsed["feedback"], "raw": raw})
        print(f"[{i + 1}/{len(ds.tasks[3])}] {s['sample_id']} | {parsed['feedback'][:80]}", flush=True)
    result = score(name, ds, records)
    (EVAL_DIR / f"{name}.json").write_text(json.dumps({"metrics": result, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen-targets", "train", "eval"])
    ap.add_argument("--teacher", default=TEACHER_MODEL, help="gen-targets: model that writes the feedback targets")
    ap.add_argument("--num-shards", type=int, default=1, help="gen-targets: split samples across processes/GPUs")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="gen-targets: only first N synth samples")
    ap.add_argument("--per-bug", type=int, default=None, help="gen-targets: only N samples per (problem, bug type) -- quick local subset")
    ap.add_argument("--model", default=MODEL_NAME, help="train/eval: student model (adapter saved per model)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help="train: stop after N optimizer steps (smoke test)")
    ap.add_argument("--base", action="store_true", help="eval: no adapter, same prompt")
    args = ap.parse_args()
    if args.step == "gen-targets":
        gen_targets(args.teacher, args.num_shards, args.shard_id, args.limit, args.per_bug)
    elif args.step == "train":
        train(args.model, args.epochs, args.lr, args.max_steps)
    else:
        evaluate(args.model, args.base)
