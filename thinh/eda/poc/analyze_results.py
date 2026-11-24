"""Deep-dive statistics behind the POC headline numbers, written as one JSON pack.

Answers: where the pooled QWK comes from (easy compile failures, exam identity), how stable the
EX01 LLM grades are across runs, how sensitive the EX02 rule mapping is, how much the label fixes
favour signal-based methods, what limits macro-F1, and whether differences survive a paired bootstrap.
Usage: python poc/analyze_results.py --data-root <dataset> --preds-dir <dir> --out pack.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import DIMS, load_meta, load_samples
from score_predictions import gold_table
from signals import compile_status, ex01_functions, statement_weights, test_summary

AXIS = list(range(11))
HEADLINE = "hybrid-v2-vote2"
EX01_RUNS = ["opus-code-only+gate", "opus-with-signals+gate", "haiku-with-signals+gate", "opus-v2+gate", "haiku-v2+gate"]
COMPARE = ["b1-signals+rules", "opus-with-signals+gate", "opus-v2+gate", "haiku-v2+gate", "hybrid-opus"]
BOOTSTRAP = 4000
# Every end-to-end configuration that was scored, i.e. the pool the headline was picked from
# (hybrid-opus-v2+gate is left out: its totals are identical to the headline's).
SELECTION_POOL = ["b1-signals", "b1-signals+rules", "opus-code-only", "opus-with-signals", "opus-with-signals+gate",
                  "opus-with-signals+rules", "haiku-with-signals", "haiku-with-signals+gate", "opus-v2", "opus-v2+gate",
                  "haiku-v2", "haiku-v2+gate", "hybrid-opus", "hybrid-haiku-v2+gate", HEADLINE]


def qwk(y, p) -> float:
    return round(float(cohen_kappa_score(y, p, weights="quadratic", labels=AXIS)), 3)


def load_preds(preds_dir: Path, name: str) -> dict[str, dict]:
    lines = (preds_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return {p["sample_id"]: p for p in map(json.loads, lines)}


def totals(gold: dict, preds: dict, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([gold[i]["total"] for i in ids]), np.array([int(preds[i]["total_score"]) for i in ids])


def label_matrix(rows: dict, ids: list[str], taxonomy: list[str], key: str) -> np.ndarray:
    return np.array([[lab in rows[i][key] for lab in taxonomy] for i in ids], dtype=int)


def ex02_rule_variant(r, logic_from: str, io_from: str) -> int:
    """Total score of the EX02 rule mapping under one design choice (compile gate always on)."""
    if compile_status(r.compile_log or "") in ("error", "link"):
        return 1
    ts = test_summary(r.test_report if isinstance(r.test_report, list) else None)
    if ts is None:
        return 10
    passed = ts["lenient_pass"] if logic_from in ("lenient", "ratio") else ts["strict_pass"]
    if logic_from == "ratio":
        logic = round(4 * passed / ts["n"])
    else:
        logic = 4 if passed == ts["n"] else (2 if passed > 0 else 0)
    io = int(ts["strict_pass"] > 0) if io_from == "any_strict_pass" else int(ts["strict_pass"] == ts["n"])
    complexity = 1  # released EX02 prior: 13/17 samples have complexity 1
    return 1 + io + logic + logic // 2 + complexity + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--preds-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    label_space, exams, _ = load_meta(args.data_root)
    taxonomy = label_space["taxonomy_error"]
    # Point weights as written in the EX01 statement (problems[].max_score says 2.5 each).
    ex01_weights = statement_weights(exams["EX01"]["statement"])
    df = load_samples(args.data_root)
    built = {r.sample_id: compile_status(r.compile_log or "") not in ("error", "link") for r in df.itertuples()}
    golds = {"fixed": gold_table(args.data_root, True), "released": gold_table(args.data_root, False)}
    gold = golds["fixed"]
    ids = sorted(gold)
    scope = {"all": ids, "EX01": [i for i in ids if i.startswith("EX01")], "EX02": [i for i in ids if i.startswith("EX02")]}
    head = load_preds(args.preds_dir, HEADLINE)
    pack: dict = {}

    # 1. Headline per scope, both gold versions, plus exact-total and within-1 rates.
    pack["headline"] = {}
    for gname, g in golds.items():
        for sname, sids in scope.items():
            y, p = totals(g, head, sids)
            Y, P = label_matrix(g, sids, taxonomy, "labels"), label_matrix(head, sids, taxonomy, "taxonomy_error")
            pack["headline"][f"{gname}|{sname}"] = {
                "n": len(sids), "qwk": qwk(y, p), "mae": round(float(np.abs(y - p).mean()), 2),
                "bias": round(float((p - y).mean()), 2), "exact_total": round(float((y == p).mean()), 3),
                "within_1": round(float((np.abs(y - p) <= 1).mean()), 3),
                "macro_f1": round(float(f1_score(Y, P, average="macro", zero_division=0)), 3),
                "micro_f1": round(float(f1_score(Y, P, average="micro", zero_division=0)), 3),
            }

    # 2. Absolute error histogram per exam (fixed gold).
    pack["abs_error_hist"] = {}
    for sname in ("EX01", "EX02"):
        y, p = totals(gold, head, scope[sname])
        err = np.abs(y - p)
        pack["abs_error_hist"][sname] = {"0": int((err == 0).sum()), "1": int((err == 1).sum()),
                                         "2": int((err == 2).sum()), ">=3": int((err >= 3).sum())}

    # 3. Easy-sample share: compile failures are decided by a rule and are near-trivial.
    hard = [i for i in ids if built[i]]
    pack["easy_vs_hard"] = {"n_compile_fail": len(ids) - len(hard), "n_built": len(hard)}
    for name in [HEADLINE, *COMPARE]:
        pr = load_preds(args.preds_dir, name)
        row = {}
        for sname, sids in (("built_all", hard), ("built_EX01", [i for i in hard if i.startswith("EX01")]),
                            ("built_EX02", [i for i in hard if i.startswith("EX02")])):
            y, p = totals(gold, pr, sids)
            row[sname] = {"n": len(sids), "qwk": qwk(y, p), "mae": round(float(np.abs(y - p).mean()), 2)}
        pack["easy_vs_hard"][name] = row

    # 4. What knowing only the exam (and the compile gate) is worth in pooled QWK.
    def exam_mean_pred(i: str, use_gate: bool) -> int:
        exam = i[:4]
        if use_gate and not built[i]:
            return 0 if exam == "EX01" else 1
        pool = [gold[j]["total"] for j in ids if j[:4] == exam and j != i and (not use_gate or built[j])]
        return int(round(float(np.mean(pool))))
    pack["exam_identity_baseline"] = {}
    for use_gate in (False, True):
        y = np.array([gold[i]["total"] for i in ids])
        p = np.array([exam_mean_pred(i, use_gate) for i in ids])
        pack["exam_identity_baseline"]["gate+exam_mean" if use_gate else "exam_mean"] = {
            "qwk_all": qwk(y, p), "mae_all": round(float(np.abs(y - p).mean()), 2)}

    # 5. EX01 run-to-run agreement vs agreement with gold; simple ensembles of totals.
    ex01 = scope["EX01"]
    runs = {n: load_preds(args.preds_dir, n) for n in EX01_RUNS}
    vec = {n: np.array([int(r[i]["total_score"]) for i in ex01]) for n, r in runs.items()}
    y01 = np.array([gold[i]["total"] for i in ex01])
    pack["ex01_agreement"] = {
        "vs_gold": {n: qwk(y01, v) for n, v in vec.items()},
        "pairwise": {f"{a}~{b}": qwk(vec[a], vec[b]) for a, b in itertools.combinations(EX01_RUNS, 2)},
    }
    stack = np.vstack([vec[n] for n in EX01_RUNS if n != "opus-code-only+gate"])
    pack["ex01_agreement"]["ensemble_vs_gold"] = {
        "mean_round": qwk(y01, np.rint(stack.mean(0)).astype(int)), "median": qwk(y01, np.rint(np.median(stack, 0)).astype(int))}
    pack["ex01_agreement"]["per_sample"] = {
        i: {"gold": int(y01[k]), **{n: int(vec[n][k]) for n in EX01_RUNS}, "spread": int(stack[:, k].max() - stack[:, k].min())}
        for k, i in enumerate(ex01)}

    # 6. EX01 per-dimension accuracy of the headline (its EX01 rubric = opus-v2+gate).
    pack["ex01_dims"] = {}
    for d in DIMS:
        y = np.array([gold[i]["rubric"][d] for i in ex01])
        p = np.array([head[i]["rubric"][d] for i in ex01])
        pack["ex01_dims"][d] = {"exact": round(float((y == p).mean()), 3), "mae": round(float(np.abs(y - p).mean()), 2),
                                "bias": round(float((p - y).mean()), 2)}

    # 7. EX02 rule-mapping sensitivity, on both gold versions.
    rows02 = [r for r in df.sort_values("sample_id").itertuples() if r.exam_id == "EX02"]
    pack["ex02_rule_sensitivity"] = {}
    for logic_from, io_from in itertools.product(("lenient", "strict", "ratio"), ("any_strict_pass", "all_strict_pass")):
        p = np.array([ex02_rule_variant(r, logic_from, io_from) for r in rows02])
        pack["ex02_rule_sensitivity"][f"logic={logic_from}|io={io_from}"] = {
            g: qwk(np.array([golds[g][r.sample_id]["total"] for r in rows02]), p) for g in golds}

    # 8. Circularity check: how much each method gains from the label fixes, per exam.
    pack["label_fix_gain"] = {}
    for name in [HEADLINE, *COMPARE]:
        pr = load_preds(args.preds_dir, name)
        row = {}
        for sname in ("EX02", "all"):
            q = {g: qwk(*totals(golds[g], pr, scope[sname])) for g in golds}
            row[sname] = {"released": q["released"], "fixed": q["fixed"], "gain": round(q["fixed"] - q["released"], 3)}
        pack["label_fix_gain"][name] = row

    # 9. Macro-F1 anatomy for the headline: per-label support, precision, recall, F1.
    Y, P = label_matrix(gold, ids, taxonomy, "labels"), label_matrix(head, ids, taxonomy, "taxonomy_error")
    per = {}
    for k, lab in enumerate(taxonomy):
        per[lab] = {"support": int(Y[:, k].sum()), "predicted": int(P[:, k].sum()),
                    "precision": round(float(precision_score(Y[:, k], P[:, k], zero_division=0)), 3),
                    "recall": round(float(recall_score(Y[:, k], P[:, k], zero_division=0)), 3),
                    "f1": round(float(f1_score(Y[:, k], P[:, k], zero_division=0)), 3)}
    supported = [k for k, lab in enumerate(taxonomy) if per[lab]["support"] > 0]
    pack["macro_f1_anatomy"] = {
        "per_label": per,
        "macro_f1_zero_division_0": round(float(f1_score(Y, P, average="macro", zero_division=0)), 3),
        "macro_f1_zero_division_1": round(float(f1_score(Y, P, average="macro", zero_division=1)), 3),
        "macro_f1_supported_labels_only": round(float(f1_score(Y[:, supported], P[:, supported], average="macro", zero_division=0)), 3),
        "ceiling_zero_division_0": round(len(supported) / len(taxonomy), 2),
        "empty_list_samples_gold": int((Y.sum(1) == 0).sum()), "empty_list_samples_pred": int((P.sum(1) == 0).sum()),
    }

    # 10. Paired bootstrap of the headline against each comparison run (fixed gold, all samples).
    rng = np.random.default_rng(7)
    yh, ph = totals(gold, head, ids)
    Ph = label_matrix(head, ids, taxonomy, "taxonomy_error")
    pack["paired_bootstrap"] = {}
    for name in COMPARE:
        pr = load_preds(args.preds_dir, name)
        _, pc = totals(gold, pr, ids)
        Pc = label_matrix(pr, ids, taxonomy, "taxonomy_error")
        dq, df1 = [], []
        for _ in range(BOOTSTRAP):
            b = rng.integers(0, len(ids), len(ids))
            dq.append(cohen_kappa_score(yh[b], ph[b], weights="quadratic", labels=AXIS)
                      - cohen_kappa_score(yh[b], pc[b], weights="quadratic", labels=AXIS))
            df1.append(f1_score(Y[b], Ph[b], average="macro", zero_division=0) - f1_score(Y[b], Pc[b], average="macro", zero_division=0))
        pack["paired_bootstrap"][name] = {
            "qwk_diff": round(qwk(yh, ph) - qwk(yh, pc), 3), "qwk_diff_ci95": [round(float(v), 3) for v in np.percentile(dq, [2.5, 97.5])],
            "p_headline_better_qwk": round(float(np.mean(np.array(dq) > 0)), 3),
            "f1_diff_ci95": [round(float(v), 3) for v in np.percentile(df1, [2.5, 97.5])],
            "p_headline_better_f1": round(float(np.mean(np.array(df1) > 0)), 3)}

    # 11. Leave-suspects-out: EX01 QWK without samples that look like label noise.
    pack["ex01_without_suspects"] = {}
    for drop in (["EX01-S112"], ["EX01-S112", "EX01-S110", "EX01-S108"]):
        keep = [i for i in ex01 if i not in drop]
        pack["ex01_without_suspects"]["-" + ",".join(s[-4:] for s in drop)] = qwk(*totals(gold, head, keep))

    # 12. Selection optimism with a control: OOB of the selected config minus OOB of the fixed headline
    # on the same draws isolates the selection effect from small-sample QWK bias. Run on both golds.
    cands = {n: load_preds(args.preds_dir, n) for n in SELECTION_POOL}
    pack["selection_optimism"] = {"n_configs": len(cands)}
    for gname, g in golds.items():
        y_all = np.array([g[i]["total"] for i in ids])
        p_all = {n: np.array([int(c[i]["total_score"]) for i in ids]) for n, c in cands.items()}
        in_bag, oob_sel, oob_fix, picks = [], [], [], []
        for _ in range(1000):
            b = rng.integers(0, len(ids), len(ids))
            out = np.setdiff1d(np.arange(len(ids)), b)
            if len(set(y_all[out])) < 2:
                continue
            scores = {n: cohen_kappa_score(y_all[b], p[b], weights="quadratic", labels=AXIS) for n, p in p_all.items()}
            best = max(scores, key=scores.get)
            picks.append(best)
            in_bag.append(scores[best])
            oob_sel.append(cohen_kappa_score(y_all[out], p_all[best][out], weights="quadratic", labels=AXIS))
            oob_fix.append(cohen_kappa_score(y_all[out], p_all[HEADLINE][out], weights="quadratic", labels=AXIS))
        gap = np.array(oob_sel) - np.array(oob_fix)
        pack["selection_optimism"][gname] = {
            "full_sample_headline": qwk(y_all, p_all[HEADLINE]), "in_bag_best_mean": round(float(np.mean(in_bag)), 3),
            "oob_selected_mean": round(float(np.mean(oob_sel)), 3), "oob_fixed_headline_mean": round(float(np.mean(oob_fix)), 3),
            "selection_effect_mean": round(float(gap.mean()), 3),
            "pick_share": {n: round(picks.count(n) / len(picks), 3) for n in sorted(set(picks), key=picks.count, reverse=True)[:4]}}

    # 13. Metric-convention sensitivity: never predicting labels absent from gold, under both zero_division values.
    absent = [k for k, lab in enumerate(taxonomy) if per[lab]["support"] == 0]
    P_trim = P.copy()
    P_trim[:, absent] = 0
    pack["macro_f1_anatomy"]["if_absent_labels_never_predicted"] = {
        "zero_division_0": round(float(f1_score(Y, P_trim, average="macro", zero_division=0)), 3),
        "zero_division_1": round(float(f1_score(Y, P_trim, average="macro", zero_division=1)), 3)}

    # 14. Fair no-LLM baselines (training means are in-sample, so slightly optimistic, like the rules).
    b1 = load_preds(args.preds_dir, "b1-signals+rules")
    built_mean = {ex: int(round(float(np.mean([gold[i]["total"] for i in scope[ex] if built[i]])))) for ex in ("EX01", "EX02")}
    fail_total = {"EX01": 0, "EX02": 1}
    baselines = {
        "gate+train_mean": {i: (built_mean[i[:4]] if built[i] else fail_total[i[:4]]) for i in ids},
        "gate+EX01_mean+EX02_rule": {i: (b1[i]["total_score"] if i.startswith("EX02") else (built_mean["EX01"] if built[i] else 0)) for i in ids},
    }
    # EX01 statement-weight predictor: sum of P1..P4 weights for functions that have a body (no gold, no LLM).
    code = {r.sample_id: r.code or "" for r in df.itertuples()}
    presence = {i: (round(sum(w for k, w in ex01_weights.items() if ex01_functions(code[i])[k] == "defined")) if built[i] else 0)
                for i in scope["EX01"]}
    baselines["gate+EX01_presence_weights+EX02_rule"] = {i: (b1[i]["total_score"] if i.startswith("EX02") else presence[i]) for i in ids}
    pack["fair_baselines"] = {"ex01_built_mean": built_mean["EX01"], "ex02_built_mean": built_mean["EX02"]}
    for name, pred in baselines.items():
        row = {}
        for sname, sids in (("all", ids), ("EX01", scope["EX01"]), ("EX01_built", [i for i in scope["EX01"] if built[i]])):
            y = np.array([gold[i]["total"] for i in sids]); p = np.array([pred[i] for i in sids])
            row[sname] = {"qwk": qwk(y, p), "mae": round(float(np.abs(y - p).mean()), 2), "within_1": int((np.abs(y - p) <= 1).sum()), "n": len(sids)}
        dq = []
        p_base = np.array([pred[i] for i in ids])
        for _ in range(BOOTSTRAP):
            b = rng.integers(0, len(ids), len(ids))
            dq.append(cohen_kappa_score(yh[b], ph[b], weights="quadratic", labels=AXIS) - cohen_kappa_score(yh[b], p_base[b], weights="quadratic", labels=AXIS))
        row["headline_minus_baseline"] = {"qwk_diff": round(qwk(yh, ph) - qwk(yh, p_base), 3),
                                          "ci95": [round(float(v), 3) for v in np.percentile(dq, [2.5, 97.5])],
                                          "share_positive": round(float(np.mean(np.array(dq) > 0)), 3)}
        pack["fair_baselines"][name] = row
    pack["fair_baselines"]["ex01_presence_per_sample"] = {
        i: {"gold": gold[i]["total"], "presence_pred": presence[i], "llm": int(head[i]["total_score"]),
            "functions": {k: v for k, v in ex01_functions(code[i]).items() if k != "main"}} for i in scope["EX01"]}

    # 15. EX01 scale compression on built samples (QWK on a fixed axis equals Lin's CCC, so shrinkage is penalised).
    eb = [i for i in scope["EX01"] if built[i]]
    y, p = totals(gold, head, eb)
    pack["ex01_compression"] = {"n": len(eb), "var_gold": round(float(y.var()), 2), "var_pred": round(float(p.var()), 2),
                                "slope_pred_on_gold": round(float(np.polyfit(y, p, 1)[0]), 2),
                                "pearson": round(float(np.corrcoef(y, p)[0, 1]), 2),
                                "gold_values": sorted(int(v) for v in y), "pred_values": sorted(int(v) for v in p)}

    args.out.write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
