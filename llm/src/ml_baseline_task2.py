"""Classical ML baseline for Task 2 (taxonomy classification) -- NOT a replacement for the
required LLM approaches (the course rules require >=1 prompting + >=1 fine-tuned open-weight
LLM as the graded deliverables), just an additional comparison point for the report's
ablation/error-analysis section. Code embeddings (sentence-transformers, CPU) + a simple
multi-label logistic regression -- deliberately simple given the tiny (89-sample) pool.
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from data import REPO_ROOT, load_public_test, read_code
from eval import task2_metrics
from rag import EMBED_MODEL, load_pool

SEED = 42
OUTPUT_DIR = REPO_ROOT / "outputs"


def build_training_table(task_num=2):
    pool = load_pool(task_num)
    codes = [read_code(ds, s) for ds, s in pool]
    labels = [s["output"]["taxonomy_error"] for _, s in pool]
    sample_ids = [s["sample_id"] for _, s in pool]
    return pool, codes, labels, sample_ids


def main():
    label_space_ds = load_pool(2)[0][0]  # any pool item's Dataset has the shared label_space
    label_space = label_space_ds.label_space["taxonomy_error"]
    mlb = MultiLabelBinarizer(classes=label_space)

    pool, codes, labels, sample_ids = build_training_table()
    print(f"training pool: {len(pool)} samples")

    print(f"embedding with {EMBED_MODEL} (CPU)...")
    model = SentenceTransformer(EMBED_MODEL)
    X = model.encode(codes, normalize_embeddings=True)
    Y = mlb.fit_transform(labels)

    # 5-fold out-of-fold CV for an honest estimate on this tiny pool
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred = np.zeros_like(Y)
    for train_idx, val_idx in kf.split(X):
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
        clf.fit(X[train_idx], Y[train_idx])
        oof_pred[val_idx] = clf.predict(X[val_idx])

    gold_samples = [{"sample_id": sid, "output": {"taxonomy_error": lbl}} for sid, lbl in zip(sample_ids, labels)]
    pred_by_id = {sid: mlb.inverse_transform(oof_pred[i:i+1])[0] for i, sid in enumerate(sample_ids)}
    pred_by_id = {sid: list(v) for sid, v in pred_by_id.items()}
    m = task2_metrics(gold_samples, pred_by_id, label_space)
    print(f"5-fold OOF CV: macro_f1={m['macro_f1']:.3f} micro_f1={m['micro_f1']:.3f}")
    print("per-label F1:", {k: round(v, 2) for k, v in m["per_label_f1"].items()})

    # Train final model on ALL 89 samples, predict on the real public test set
    final_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    final_clf.fit(X, Y)

    ds_public = load_public_test()
    public_samples = ds_public.tasks[2]
    public_codes = [read_code(ds_public, s) for s in public_samples]
    X_public = model.encode(public_codes, normalize_embeddings=True)
    pred_public = final_clf.predict(X_public)
    pred_labels = mlb.inverse_transform(pred_public)

    predictions = [
        {"sample_id": s["sample_id"], "output": {"taxonomy_error": list(labels_i)}}
        for s, labels_i in zip(public_samples, pred_labels)
    ]
    out_path = OUTPUT_DIR / "predictions_task2_public_mlbaseline.json"
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(predictions)} samples)")


if __name__ == "__main__":
    main()
