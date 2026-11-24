"""Validate the contract files in docs/contracts against real data, examples and negative cases.

Checks:
  1. every schema is a valid draft 2020-12 schema; every $ref resolves (local registry, no network)
  2. the dataset files of a data root validate against the observed dataset schemas
  3. every example in contracts/examples validates against its schema / OpenAPI component
  4. negative cases are rejected (feedback firewall, NFD label, out-of-range score, ...)
  5. openapi.yaml passes openapi-spec-validator
Usage: python validate_contracts.py --data-root <dataset dir> [--contracts docs/contracts]
"""

from __future__ import annotations

import argparse
import copy
import json
import unicodedata
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

BASE = "https://grader.invalid/contracts/"
DATASET_FILES = {
    "task1_grading.json": "dataset/task1-grading.schema.json",
    "task2_error_taxonomy.json": "dataset/task2-error-taxonomy.schema.json",
    "task3_feedback.json": "dataset/task3-feedback.schema.json",
    "exams.json": "dataset/exams.schema.json",
    "label_space.json": "dataset/label-space.schema.json",
}
# example file stem -> schema URI (relative to BASE) or OpenAPI component pointer
EXAMPLE_TARGETS = {
    "predictions-t1": "predictions/predictions-t1.schema.json",
    "predictions-t2": "predictions/predictions-t2.schema.json",
    "predictions-t3": "predictions/predictions-t3.schema.json",
    "evidence-card-ex01": "internal/evidence-card.schema.json",
    "evidence-card-ex02": "internal/evidence-card.schema.json",
    "problem-card-ex01": "internal/problem-card.schema.json",
    "run-result-quota": "internal/run-result.schema.json",
    "api-healthz-response": "openapi.yaml#/components/schemas/HealthResponse",
    "api-info-response": "openapi.yaml#/components/schemas/InfoResponse",
    "api-evidence-request": "openapi.yaml#/components/schemas/BlindRequest",
    "api-evidence-response": "openapi.yaml#/components/schemas/EvidenceResponse",
    "api-grade-request": "openapi.yaml#/components/schemas/BlindRequest",
    "api-grade-response": "openapi.yaml#/components/schemas/GradeResponse",
    "api-classify-request": "openapi.yaml#/components/schemas/BlindRequest",
    "api-classify-response": "openapi.yaml#/components/schemas/ClassifyResponse",
    "api-feedback-request": "openapi.yaml#/components/schemas/FeedbackRequest",
    "api-feedback-response": "openapi.yaml#/components/schemas/FeedbackResponse",
    "api-assess-request": "openapi.yaml#/components/schemas/AssessRequest",
    "api-assess-response": "openapi.yaml#/components/schemas/AssessResponse",
    "api-problem-422": "openapi.yaml#/components/schemas/Problem",
}


def build_registry(contracts: Path) -> Registry:
    resources = []
    for path in sorted(contracts.rglob("*.schema.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        uri = doc.get("$id") or BASE + path.relative_to(contracts).as_posix()
        resources.append((uri, Resource.from_contents(doc, default_specification=DRAFT202012)))
    openapi = yaml.safe_load((contracts / "openapi.yaml").read_text(encoding="utf-8"))
    resources.append((BASE + "openapi.yaml", Resource(contents=openapi, specification=DRAFT202012)))
    return Registry().with_resources(resources)


def validator_for(target: str, registry: Registry) -> Draft202012Validator:
    return Draft202012Validator({"$ref": BASE + target}, registry=registry)


def errors(target: str, instance: object, registry: Registry) -> list[str]:
    found = validator_for(target, registry).iter_errors(instance)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:160]}" for e in found]


def check_schemas(contracts: Path, registry: Registry) -> int:
    bad = 0
    for path in sorted(contracts.rglob("*.schema.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        try:
            Draft202012Validator.check_schema(doc)
            # resolve every $ref by validating a trivial instance through the registry
            list(validator_for(path.relative_to(contracts).as_posix(), registry).iter_errors({}))
        except Exception as exc:  # report and continue: this is a checker, not a pipeline stage
            bad += 1
            print(f"  FAIL schema {path.name}: {type(exc).__name__}: {str(exc)[:200]}")
    return bad


def check_dataset(root: Path, registry: Registry) -> int:
    bad = 0
    for name, target in DATASET_FILES.items():
        errs = errors(target, json.loads((root / name).read_text(encoding="utf-8")), registry)
        bad += bool(errs)
        print(f"  {'OK  ' if not errs else 'FAIL'} {name} -> {target} ({len(errs)} errors)")
        for e in errs[:8]:
            print(f"       {e}")
    return bad


def check_examples(contracts: Path, registry: Registry) -> int:
    bad = 0
    seen = set()
    for path in sorted((contracts / "examples").glob("*.json")):
        target = EXAMPLE_TARGETS.get(path.stem)
        seen.add(path.stem)
        if target is None:
            print(f"  SKIP {path.name}: no target mapped")
            continue
        errs = errors(target, json.loads(path.read_text(encoding="utf-8")), registry)
        bad += bool(errs)
        print(f"  {'OK  ' if not errs else 'FAIL'} {path.name} -> {target} ({len(errs)} errors)")
        for e in errs[:6]:
            print(f"       {e}")
    for stem in sorted(set(EXAMPLE_TARGETS) - seen):
        print(f"  MISSING example {stem}.json")
    return bad


def negative_cases(root: Path, contracts: Path, registry: Registry) -> int:
    """Each mutation must be REJECTED; returns how many slipped through."""
    t1 = json.loads((root / "task1_grading.json").read_text(encoding="utf-8"))
    t2 = json.loads((root / "task2_error_taxonomy.json").read_text(encoding="utf-8"))
    t3 = json.loads((root / "task3_feedback.json").read_text(encoding="utf-8"))
    ex = contracts / "examples"
    grade_req = json.loads((ex / "api-grade-request.json").read_text(encoding="utf-8"))
    pred_t2 = json.loads((ex / "predictions-t2.json").read_text(encoding="utf-8"))
    pred_t1 = json.loads((ex / "predictions-t1.json").read_text(encoding="utf-8"))

    def mutate(doc: dict, fn) -> dict:
        out = copy.deepcopy(doc)
        fn(out)
        return out

    def first_item(doc):  # predictions examples may be a list or an envelope with "predictions"
        return doc[0] if isinstance(doc, list) else next(v for v in doc.values() if isinstance(v, list))[0]

    nfd = unicodedata.normalize("NFD", "Lỗi logic")
    cases = [
        ("T1 total_score=11", "dataset/task1-grading.schema.json",
         mutate(t1, lambda d: d["samples"][0]["output"].update(total_score=11))),
        ("T1 logic=5", "dataset/task1-grading.schema.json",
         mutate(t1, lambda d: d["samples"][0]["output"]["rubric"].update(logic=5))),
        ("T1 input carries feedback", "dataset/task1-grading.schema.json",
         mutate(t1, lambda d: d["samples"][0]["input"].update(feedback="x"))),
        ("T2 NFD label", "dataset/task2-error-taxonomy.schema.json",
         mutate(t2, lambda d: d["samples"][0]["output"].update(taxonomy_error=[nfd]))),
        ("T2 unknown label", "dataset/task2-error-taxonomy.schema.json",
         mutate(t2, lambda d: d["samples"][0]["output"].update(taxonomy_error=["Lỗi khác"]))),
        ("T3 input with test_report", "dataset/task3-feedback.schema.json",
         mutate(t3, lambda d: d["samples"][0]["input"].update(test_report=[]))),
        ("T3 bad level string", "dataset/task3-feedback.schema.json",
         mutate(t3, lambda d: d["samples"][0]["input"].update(target_feedback_level="Level 5"))),
        ("API grade request with feedback", "openapi.yaml#/components/schemas/BlindRequest",
         mutate(grade_req, lambda d: d.update(feedback="x"))),
        ("API grade request with target level", "openapi.yaml#/components/schemas/BlindRequest",
         mutate(grade_req, lambda d: d.update(target_feedback_level="Level 1 - Gợi ý nhẹ"))),
        ("pred T2 NFD label", "predictions/predictions-t2.schema.json",
         mutate(pred_t2, lambda d: first_item(d).__setitem__("taxonomy_error", [nfd])
                if "taxonomy_error" in first_item(d) else first_item(d)["output"].__setitem__("taxonomy_error", [nfd]))),
        ("pred T1 total 10.5", "predictions/predictions-t1.schema.json",
         mutate(pred_t1, lambda d: (first_item(d).get("output", first_item(d))).__setitem__("total_score", 10.5))),
    ]
    slipped = 0
    for label, target, inst in cases:
        rejected = bool(errors(target, inst, registry))
        slipped += not rejected
        print(f"  {'OK  ' if rejected else 'MISS'} rejects: {label}")
    return slipped


def check_openapi(contracts: Path) -> int:
    from openapi_spec_validator import validate
    from openapi_spec_validator.readers import read_from_filename

    spec, base_uri = read_from_filename(str(contracts / "openapi.yaml"))
    try:
        validate(spec, base_uri=base_uri)
        print("  OK   openapi.yaml (openapi-spec-validator, external $refs resolved from disk)")
        return 0
    except Exception as exc:
        print(f"  FAIL openapi.yaml: {type(exc).__name__}: {str(exc)[:400]}")
        return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--contracts", type=Path, default=Path(__file__).resolve().parent / "docs" / "contracts")
    args = ap.parse_args()
    registry = build_registry(args.contracts)
    results = {}
    print("1. schemas (meta-schema + $ref resolution)")
    results["schemas"] = check_schemas(args.contracts, registry)
    print("2. dataset files vs observed schemas")
    results["dataset"] = check_dataset(args.data_root, registry)
    print("3. examples")
    results["examples"] = check_examples(args.contracts, registry)
    print("4. negative cases (must be rejected)")
    results["negatives_slipped"] = negative_cases(args.data_root, args.contracts, registry)
    print("5. OpenAPI")
    results["openapi"] = check_openapi(args.contracts)
    print("SUMMARY", results)
    raise SystemExit(1 if any(results.values()) else 0)


if __name__ == "__main__":
    main()
