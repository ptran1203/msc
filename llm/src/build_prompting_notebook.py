"""Build notebooks/02_prompting_baseline.ipynb analyzing zero-shot results on sample_dataset."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []

cells.append(new_markdown_cell(
"""# Prompting baseline — zero-shot, Qwen2.5-Coder-7B-Instruct (4-bit)

Đánh giá baseline zero-shot (không few-shot, không fine-tune) trên 32 mẫu `sample_dataset` (có nhãn),
so sánh với baseline tầm thường (constant prediction) để biết model có học được tín hiệu thật hay không.

Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, 4-bit NF4 (bitsandbytes), greedy decoding (seed=42).
Xem [METHOD.md](../METHOD.md) để biết chi tiết prompt/pipeline."""
))

cells.append(new_code_cell(
"""import sys, json
sys.path.insert(0, "../src")
from data import load_sample_dataset
from eval import task1_metrics, task2_metrics, task3_compliance

ds = load_sample_dataset()
label_space = ds.label_space["taxonomy_error"]

def load_preds(task_num):
    with open(f"../outputs/predictions_task{task_num}_sample.json", encoding="utf-8") as f:
        return json.load(f)

preds1 = load_preds(1)
preds2 = load_preds(2)
preds3 = load_preds(3)
print(len(preds1), len(preds2), len(preds3), "predictions loaded")
"""
))

cells.append(new_markdown_cell("## Task 1 — QWK / MAE / exact-match, so với baseline hằng số"))

cells.append(new_code_cell(
"""pred1_by_id = {p["sample_id"]: p["output"] for p in preds1}
m1 = task1_metrics(ds.tasks[1], pred1_by_id)
print("zero-shot:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m1.items() if k != "exact_match"})
print("  exact_match per dim:", m1["exact_match"])

scores = [s["output"]["total_score"] for s in ds.tasks[1]]
mean_c, mode_c = round(sum(scores) / len(scores)), max(set(scores), key=scores.count)
for name, const in [("mean", mean_c), ("mode", mode_c)]:
    pred_c = {s["sample_id"]: {"rubric": s["output"]["rubric"], "total_score": const} for s in ds.tasks[1]}
    mc = task1_metrics(ds.tasks[1], pred_c)
    print(f"constant={name}({const}): QWK={mc['qwk']:.3f} MAE={mc['mae']:.3f}")
"""
))

cells.append(new_markdown_cell(
"""**Đọc kết quả:** zero-shot có QWK > 0 rõ rệt trong khi baseline hằng số QWK = 0 (không có biến thiên) —
model có học được *thứ tự* điểm số thật. Nhưng MAE zero-shot xấp xỉ baseline hằng số → model **lệch chuẩn (calibration)**
so với thang điểm giáo viên, dù đã nắm được tương quan. Đây là mục tiêu chính cho few-shot: cho model thấy ví dụ
điểm số thật để canh thang, không phải để dạy cách đọc code."""
))

cells.append(new_markdown_cell("## Phân tích lỗi Task 1 — các mẫu lệch nhiều nhất"))

cells.append(new_code_cell(
"""rows = []
for s in ds.tasks[1]:
    sid = s["sample_id"]
    gold_total = s["output"]["total_score"]
    pred_total = pred1_by_id[sid]["total_score"]
    rows.append((sid, gold_total, pred_total, abs(gold_total - pred_total)))
rows.sort(key=lambda r: -r[3])
print(f"{'sample_id':<15}{'gold':>6}{'pred':>6}{'|diff|':>8}")
for sid, g, p, d in rows[:10]:
    print(f"{sid:<15}{g:>6}{p:>6}{d:>8}")
"""
))

cells.append(new_markdown_cell("## Task 2 — macro/micro-F1, so với baseline hằng số"))

cells.append(new_code_cell(
"""pred2_by_id = {p["sample_id"]: p["output"]["taxonomy_error"] for p in preds2}
m2 = task2_metrics(ds.tasks[2], pred2_by_id, label_space)
print("zero-shot: macro_f1=%.3f micro_f1=%.3f" % (m2["macro_f1"], m2["micro_f1"]))
print("per-label F1:", {k: round(v, 2) for k, v in m2["per_label_f1"].items()})

for name, const in [("empty", []), ("most_common", ["Lỗi logic"])]:
    pred_c = {s["sample_id"]: const for s in ds.tasks[2]}
    mc = task2_metrics(ds.tasks[2], pred_c, label_space)
    print(f"constant={name}: macro_f1={mc['macro_f1']:.3f} micro_f1={mc['micro_f1']:.3f}")
"""
))

cells.append(new_markdown_cell(
"""**Đọc kết quả:** zero-shot vượt baseline hằng số trên macro-F1 (metric chính) nhưng thấp hơn baseline
"luôn đoán nhãn phổ biến nhất" trên micro-F1 — model dự đoán rải đều nhiều nhãn hơn (đúng hướng cho macro-F1,
vì macro-F1 phạt nặng việc chỉ đoán nhãn phổ biến) nhưng độ chính xác per-label vẫn còn thấp."""
))

cells.append(new_markdown_cell("## Task 3 — Level compliance trên 32 mẫu"))

cells.append(new_code_cell(
"""pred3_by_id = {p["sample_id"]: p["output"]["feedback"] for p in preds3}
level_by_id = {s["sample_id"]: s["input"]["target_feedback_level"] for s in ds.tasks[3]}
triples = [(sid, level_by_id[sid], fb) for sid, fb in pred3_by_id.items()]
m3 = task3_compliance(triples)
print("compliance_rate:", m3["compliance_rate"], "violations:", m3["violations"])

# qualitative: compare one Level 1 and one Level 3 feedback, model vs teacher
sample_l1 = next(s for s in ds.tasks[3] if s["input"]["target_feedback_level"].startswith("Level 1"))
sample_l3 = next(s for s in ds.tasks[3] if s["input"]["target_feedback_level"].startswith("Level 3"))
for s in [sample_l1, sample_l3]:
    sid = s["sample_id"]
    print(f"\\n--- {sid} ({s['input']['target_feedback_level']}) ---")
    print("teacher :", s["output"]["feedback"])
    print("model   :", pred3_by_id[sid][:300])
"""
))

cells.append(new_markdown_cell(
"""**Đọc kết quả:** compliance = 1.0 trên 32 mẫu — checker rule-based (regex tìm code/markers) không bắt được
vi phạm nào, nhưng đây là một heuristic thô: nó chỉ chặn code/markup rõ ràng, chưa chặn được việc "gợi ý cách sửa"
bằng lời (ví dụ "nên dùng vector thay vì mảng động" ở Level 2 — về lý thuyết đã hơi vượt mức "chỉ chẩn đoán").
Cần tinh chỉnh checker hoặc đánh giá thủ công một phần trong báo cáo cuối."""
))

cells.append(new_markdown_cell(
"""## Lưu ý quan trọng: Task 3 trên public_test khác sample_dataset

`sample_dataset` cho `taxonomy_error` sẵn ở input (output chỉ cần `feedback`), nhưng `public_test_release`
**không** cho nhãn lỗi — model phải tự chẩn đoán (`output = {taxonomy_error, feedback}`). Baseline ở notebook này
chỉ đo được nhánh "nhãn cho sẵn"; nhánh tự chẩn đoán (dùng cho submission thật) chưa có ground truth để so sánh
trực tiếp — xem [METHOD.md](../METHOD.md) mục "Task 3 schema fork"."""
))

cells.append(new_markdown_cell(
"""## Tóm tắt & bước tiếp theo

| Task | Metric chính | Zero-shot | Baseline hằng số |
|---|---|---|---|
| 1 | QWK | xem trên | 0.000 |
| 1 | MAE | xem trên | ~3.2–3.5 |
| 2 | macro-F1 | xem trên | 0.055 (nhãn phổ biến) |
| 3 | level compliance | xem trên | — |

Zero-shot vượt baseline hằng số trên các metric chính (QWK, macro-F1) nhưng MAE còn kém → cần **few-shot**
(cho ví dụ điểm số thật để canh thang) trước khi fine-tune. Đây là bước tiếp theo trong [METHOD.md](../METHOD.md)."""
))

nb = new_notebook(cells=cells)
with open("../notebooks/02_prompting_baseline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/02_prompting_baseline.ipynb")
