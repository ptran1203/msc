"""Sanity check: scoring gold labels against themselves must be perfect."""
from data import load_sample_dataset
from eval import task1_metrics, task2_metrics, task3_compliance

ds = load_sample_dataset()
label_space = ds.label_space["taxonomy_error"]

t1 = ds.tasks[1]
pred1 = {s["sample_id"]: s["output"] for s in t1}
m1 = task1_metrics(t1, pred1)
assert m1["qwk"] > 0.999, m1
assert m1["mae"] == 0.0, m1
assert all(v == 1.0 for v in m1["exact_match"].values()), m1
print("Task1 self-check OK:", m1["qwk"], m1["mae"])

t2 = ds.tasks[2]
pred2 = {s["sample_id"]: s["output"]["taxonomy_error"] for s in t2}
m2 = task2_metrics(t2, pred2, label_space)
labels_in_gold = {l for s in t2 for l in s["output"]["taxonomy_error"]}
absent_labels = set(label_space) - labels_in_gold
for label in labels_in_gold:
    assert m2["per_label_f1"][label] == 1.0, (label, m2["per_label_f1"][label])
assert m2["micro_f1"] == 1.0, m2
print("Task2 self-check OK (micro_f1=1.0; macro_f1=%.3f, deflated by %d label(s) absent from gold: %s)"
      % (m2["macro_f1"], len(absent_labels), absent_labels))

t3 = ds.tasks[3]
triples = [(s["sample_id"], s["input"]["target_feedback_level"], s["output"]["feedback"]) for s in t3]
m3 = task3_compliance(triples)
print("Task3 compliance on teacher's own feedback:", m3["compliance_rate"], "violations:", m3["violations"])

print("\nALL SANITY CHECKS PASSED")
