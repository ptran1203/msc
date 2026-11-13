"""Metrics for Task 1 (QWK/MAE/exact-match), Task 2 (micro/macro-F1), Task 3 (level compliance)."""
import re
from collections import Counter

RUBRIC_DIMS = ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]


def quadratic_weighted_kappa(y_true, y_pred, min_rating=0, max_rating=10):
    n = max_rating - min_rating + 1
    O = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        O[t - min_rating][p - min_rating] += 1
    hist_true = [sum(row) for row in O]
    hist_pred = [sum(O[i][j] for i in range(n)) for j in range(n)]
    N = len(y_true)
    E = [[hist_true[i] * hist_pred[j] / N for j in range(n)] for i in range(n)]
    W = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    num = sum(W[i][j] * O[i][j] for i in range(n) for j in range(n))
    den = sum(W[i][j] * E[i][j] for i in range(n) for j in range(n))
    if den == 0:
        return 1.0
    return 1 - num / den


def mae(y_true, y_pred):
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def task1_metrics(gold_samples, pred_by_id):
    y_true, y_pred = [], []
    exact = {d: 0 for d in RUBRIC_DIMS}
    n = 0
    for g in gold_samples:
        sid = g["sample_id"]
        if sid not in pred_by_id:
            continue
        gold = g["output"]
        pred = pred_by_id[sid]
        y_true.append(gold["total_score"])
        y_pred.append(pred["total_score"])
        for d in RUBRIC_DIMS:
            if gold["rubric"][d] == pred["rubric"].get(d):
                exact[d] += 1
        n += 1
    return {
        "n": n,
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "exact_match": {d: exact[d] / n for d in RUBRIC_DIMS} if n else {},
    }


def task2_metrics(gold_samples, pred_by_id, label_space):
    tp = Counter()
    fp = Counter()
    fn = Counter()
    n = 0
    for g in gold_samples:
        sid = g["sample_id"]
        if sid not in pred_by_id:
            continue
        gold_set = set(g["output"]["taxonomy_error"])
        pred_set = set(pred_by_id[sid])
        for label in label_space:
            in_gold, in_pred = label in gold_set, label in pred_set
            if in_gold and in_pred:
                tp[label] += 1
            elif in_pred and not in_gold:
                fp[label] += 1
            elif in_gold and not in_pred:
                fn[label] += 1
        n += 1

    def f1(label):
        p_denom = tp[label] + fp[label]
        r_denom = tp[label] + fn[label]
        precision = tp[label] / p_denom if p_denom else 0.0
        recall = tp[label] / r_denom if r_denom else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_label_f1 = {label: f1(label) for label in label_space}
    macro_f1 = sum(per_label_f1.values()) / len(label_space)
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {"n": n, "macro_f1": macro_f1, "micro_f1": micro_f1, "per_label_f1": per_label_f1}


_CODE_MARKERS = re.compile(r"(\bint\s+\w+\s*\(|#include|for\s*\(|while\s*\(|\{\s*\n|;\s*\n|```)")
_SOLUTION_WORDS = re.compile(r"(sửa thành|code sửa|lời giải|đáp án)", re.IGNORECASE)


def level_compliant(level: str, feedback: str) -> bool:
    """Level 1/2 must not contain code or an explicit fix/solution."""
    if level.startswith("Level 1") or level.startswith("Level 2"):
        if _CODE_MARKERS.search(feedback):
            return False
        if level.startswith("Level 1") and _SOLUTION_WORDS.search(feedback):
            return False
    return True


def task3_compliance(pred_samples_with_level):
    """pred_samples_with_level: list of (sample_id, level, feedback)."""
    n = len(pred_samples_with_level)
    violations = [sid for sid, level, fb in pred_samples_with_level if not level_compliant(level, fb)]
    return {"n": n, "compliance_rate": 1 - len(violations) / n if n else 1.0, "violations": violations}
