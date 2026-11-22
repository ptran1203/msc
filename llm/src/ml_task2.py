"""Task 2 classical-ML voter for the LLM ensemble (task2_from_task3.py --ml): code embeddings
+ one-vs-rest logistic regression, trained on the oracle-labeled synth set + real sample items.

Honest local estimate: leave-one-out over the 32 real items (train = all synth + the other 31
real, predict the held-out one) -> scored on real items only, like the LLM methods. The older
ml_baseline_task2.py CV (.303) mixed synth items into the validation folds, which inflates it.

Writes outputs/ml-task2-logreg/predictions_task2_{sample_loo,public}.json.
Usage: python ml_task2.py
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from data import REPO_ROOT, load_dataset, load_public_test, load_sample_dataset, read_code
from eval import task2_metrics
from rag import EMBED_MODEL, SYNTH_DIR

SEED = 42
OUT = REPO_ROOT / "outputs" / "ml-task2-logreg"


def fit(X, Y):
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    clf.fit(X, Y)
    return clf


def main():
    real = load_sample_dataset()
    synth = load_dataset(SYNTH_DIR)
    public = load_public_test()
    label_space = real.label_space["taxonomy_error"]
    mlb = MultiLabelBinarizer(classes=label_space)
    mlb.fit([label_space])

    enc = SentenceTransformer(EMBED_MODEL)
    emb = lambda ds: enc.encode([read_code(ds, s) for s in ds.tasks[2]], normalize_embeddings=True)
    Xs, Xr, Xp = emb(synth), emb(real), emb(public)
    Ys = mlb.transform([s["output"]["taxonomy_error"] for s in synth.tasks[2]])
    Yr = mlb.transform([s["output"]["taxonomy_error"] for s in real.tasks[2]])
    # labels absent from a training set break LogisticRegression (one class) -> every label
    # must have a positive and a negative somewhere; the synth set covers all 10
    assert (Ys.sum(0) > 0).all() and (Ys.sum(0) < len(Ys)).all()
    print(f"synth={len(Xs)} real={len(Xr)} public={len(Xp)} embed={EMBED_MODEL}")

    pred = {}
    for i, s in enumerate(real.tasks[2]):
        keep = np.arange(len(Xr)) != i
        clf = fit(np.vstack([Xs, Xr[keep]]), np.vstack([Ys, Yr[keep]]))
        pred[s["sample_id"]] = list(mlb.inverse_transform(clf.predict(Xr[i:i + 1]))[0])
    m = task2_metrics(real.tasks[2], pred, label_space)
    print(f"LOO on 32 real items: macro_f1={m['macro_f1']:.4f} micro_f1={m['micro_f1']:.4f}")

    clf = fit(np.vstack([Xs, Xr]), np.vstack([Ys, Yr]))
    pub = mlb.inverse_transform(clf.predict(Xp))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = lambda ids, labels: [{"sample_id": sid, "output": {"taxonomy_error": list(l)}} for sid, l in zip(ids, labels)]
    (OUT / "predictions_task2_sample_loo.json").write_text(
        json.dumps(rows(pred.keys(), pred.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "predictions_task2_public.json").write_text(
        json.dumps(rows([s["sample_id"] for s in public.tasks[2]], pub), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
