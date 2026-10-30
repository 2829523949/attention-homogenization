#!/usr/bin/env python3
"""Summarize paired facts-only vs image-plus-facts model outputs."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

CONDITIONS = ("facts_only", "image_plus_facts")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def mcnemar_exact(f_correct_vf_wrong: int, f_wrong_vf_correct: int) -> dict[str, Any]:
    b, c = int(f_correct_vf_wrong), int(f_wrong_vf_correct)
    discordant = b + c
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
        p = min(1.0, 2.0 * tail)
    return {"facts_correct_image_wrong": b, "facts_wrong_image_correct": c, "discordant": discordant, "exact_two_sided_p": p}


def metric(paired: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(paired)
    f_correct = sum(bool(row["facts_only_correct"]) for row in paired)
    vf_correct = sum(bool(row["image_plus_facts_correct"]) for row in paired)
    interference = sum(bool(row["facts_only_correct"]) and not bool(row["image_plus_facts_correct"]) for row in paired)
    rescue = sum(not bool(row["facts_only_correct"]) and bool(row["image_plus_facts_correct"]) for row in paired)
    return {
        "N": n,
        "facts_only_correct": f_correct,
        "facts_only_accuracy": f_correct / n if n else None,
        "image_plus_facts_correct": vf_correct,
        "image_plus_facts_accuracy": vf_correct / n if n else None,
        "delta_image_minus_facts": (vf_correct - f_correct) / n if n else None,
        "visual_interference_count": interference,
        "visual_interference_denominator": f_correct,
        "VIR": interference / f_correct if f_correct else None,
        "visual_rescue_count": rescue,
        "visual_rescue_denominator": n - f_correct,
        "VRR": rescue / (n - f_correct) if n - f_correct else None,
    }


def bootstrap_delta(paired: list[dict[str, Any]], seed: int, replicates: int) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_family[str(row["family_id"])].append(row)
    family_ids = sorted(by_family)
    if not family_ids:
        return {"point_estimate": None, "confidence_interval_95": [None, None], "families": 0, "valid_draws": 0, "replicates": replicates, "seed": seed}
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled_families = [rng.choice(family_ids) for _ in family_ids]
        sampled = [row for family_id in sampled_families for row in by_family[family_id]]
        value = metric(sampled)["delta_image_minus_facts"]
        if value is not None:
            draws.append(float(value))
    draws.sort()

    def percentile(q: float) -> float | None:
        if not draws:
            return None
        position = (len(draws) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return draws[lower]
        fraction = position - lower
        return draws[lower] * (1 - fraction) + draws[upper] * fraction

    return {
        "point_estimate": metric(paired)["delta_image_minus_facts"],
        "confidence_interval_95": [percentile(0.025), percentile(0.975)],
        "bootstrap_unit": "family_id",
        "families": len(family_ids),
        "valid_draws": len(draws),
        "replicates": replicates,
        "seed": seed,
    }


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in read_jsonl(path)}


def build_pairs(results: list[dict[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_case: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    duplicate_records: list[str] = []
    for row in results:
        case_id = str(row.get("case_id"))
        condition = str(row.get("condition"))
        if condition not in CONDITIONS:
            continue
        if condition in by_case[case_id]:
            duplicate_records.append(f"{case_id}::{condition}")
        by_case[case_id][condition] = row
    paired: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    ignored: list[str] = []
    for case_id, case in cases.items():
        conditions = by_case.get(case_id, {})
        if not all(condition in conditions for condition in CONDITIONS):
            incomplete.append({"case_id": case_id, "conditions_present": sorted(conditions)})
            continue
        f = conditions["facts_only"]
        vf = conditions["image_plus_facts"]
        if not bool(f.get("status") == "ok" and vf.get("status") == "ok"):
            incomplete.append({"case_id": case_id, "conditions_present": sorted(conditions), "reason": "non_ok_status"})
            continue
        paired.append(
            {
                "case_id": case_id,
                "family_id": str(case["family_id"]),
                "image_id": str(case["image_id"]),
                "qa_id": str(case["qa_id"]),
                "qa_type": str(case["qa_type"]),
                "target_side": str(case["target_side"]),
                "split": str(case["split"]),
                "target_color": str(case["target_color"]),
                "competitor_color": str(case["competitor_color"]),
                "left_is_A": bool(case["left_is_A"]),
                "same_category": bool(case["same_category"]),
                "facts_only_raw": f.get("prediction_raw"),
                "facts_only_prediction": f.get("prediction_normalized"),
                "facts_only_correct": bool(f.get("correct")),
                "facts_only_error_type": f.get("error_type"),
                "image_plus_facts_raw": vf.get("prediction_raw"),
                "image_plus_facts_prediction": vf.get("prediction_normalized"),
                "image_plus_facts_correct": bool(vf.get("correct")),
                "image_plus_facts_error_type": vf.get("error_type"),
                "transition": (
                    "BOTH_CORRECT" if bool(f.get("correct")) and bool(vf.get("correct")) else
                    "VISUAL_INTERFERENCE" if bool(f.get("correct")) and not bool(vf.get("correct")) else
                    "VISUAL_RESCUE" if not bool(f.get("correct")) and bool(vf.get("correct")) else
                    "BOTH_WRONG"
                ),
                "prompt_sha256": f.get("prompt_sha256"),
                "prompt_hash_equal": f.get("prompt_sha256") == vf.get("prompt_sha256"),
                "image_presence_contract_ok": f.get("image_present") is False and vf.get("image_present") is True,
            }
        )
    # Results not present in the frozen case manifest are diagnostic only.
    ignored = sorted(set(by_case) - set(cases))
    return paired, {
        "results_rows": len(results),
        "case_manifest_rows": len(cases),
        "paired_cases": len(paired),
        "incomplete_cases": incomplete,
        "duplicate_records": sorted(duplicate_records),
        "ignored_case_ids_not_in_manifest": ignored,
        "prompt_hash_equal_all_pairs": all(row["prompt_hash_equal"] for row in paired),
        "image_presence_contract_ok_all_pairs": all(row["image_presence_contract_ok"] for row in paired),
    }


def subset_metric(paired: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    selected = [row for row in paired if row.get(key) == value]
    return metric(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=500)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    results = read_jsonl(args.results)
    paired, audit = build_pairs(results, cases)
    all_metrics = metric(paired)
    mcnemar = mcnemar_exact(
        sum(bool(row["facts_only_correct"]) and not bool(row["image_plus_facts_correct"]) for row in paired),
        sum(not bool(row["facts_only_correct"]) and bool(row["image_plus_facts_correct"]) for row in paired),
    )
    bootstrap = bootstrap_delta(paired, args.seed, args.bootstrap_replicates)
    subsets = {
        "All": all_metrics,
        "REL2C_LEFT": subset_metric(paired, "qa_type", "REL2C_LEFT"),
        "REL2C_RIGHT": subset_metric(paired, "qa_type", "REL2C_RIGHT"),
        "same_category": subset_metric(paired, "same_category", True),
        "cross_category": subset_metric(paired, "same_category", False),
    }
    error_types = Counter(
        str(row["image_plus_facts_error_type"])
        for row in paired
        if bool(row["facts_only_correct"]) and not bool(row["image_plus_facts_correct"])
    )
    status = "COMPLETE" if len(paired) == args.expected and not audit["incomplete_cases"] else "INCOMPLETE"
    payload = {
        "schema_version": "realbind2_facts_image_500_metrics_v1",
        "status": status,
        "expected_pairs": args.expected,
        "audit": audit,
        "metrics": all_metrics,
        "subsets": subsets,
        "mcnemar_exact": mcnemar,
        "family_cluster_bootstrap_delta": bootstrap,
        "visual_interference_error_types": dict(sorted(error_types.items())),
        "transition_counts": dict(sorted(Counter(row["transition"] for row in paired).items())),
        "prompt_hash_equal_all_pairs": audit["prompt_hash_equal_all_pairs"],
        "image_presence_contract_ok_all_pairs": audit["image_presence_contract_ok_all_pairs"],
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    write_json(args.output_dir / "metrics.json", payload)
    write_jsonl(args.output_dir / "paired_cases.jsonl", paired)

    def pct(value: Any) -> str:
        return "NA" if value is None else f"{100.0 * float(value):.2f}%"

    def ci_text(value: Any) -> str:
        return "NA" if not value or value[0] is None else f"[{100.0 * value[0]:.2f}, {100.0 * value[1]:.2f}] pp"

    lines = [
        "# RealBind2 facts-only vs image-plus-facts summary",
        "",
        f"- Status: **{status}**",
        f"- Complete pairs: **{len(paired)} / {args.expected}**",
        f"- Facts-only correct: **{all_metrics['facts_only_correct']} / {all_metrics['N']}**",
        f"- Image+Facts correct: **{all_metrics['image_plus_facts_correct']} / {all_metrics['N']}**",
        f"- Δ(Image+Facts − Facts-only): **{pct(all_metrics['delta_image_minus_facts'])}**",
        f"- Family-bootstrap 95% CI: **{ci_text(bootstrap['confidence_interval_95'])}**",
        f"- VIR = P(image wrong | facts correct): **{all_metrics['visual_interference_count']} / {all_metrics['visual_interference_denominator']} = {pct(all_metrics['VIR'])}**",
        f"- McNemar exact: facts-correct/image-wrong={mcnemar['facts_correct_image_wrong']}, facts-wrong/image-correct={mcnemar['facts_wrong_image_correct']}, p={mcnemar['exact_two_sided_p']:.6g}",
        f"- Prompt hash equal for all pairs: **{audit['prompt_hash_equal_all_pairs']}**",
        f"- Image-presence contract correct for all pairs: **{audit['image_presence_contract_ok_all_pairs']}**",
        "",
        "## By side",
        "",
        "| Subset | N | Facts-only | Image+Facts | Δ | VIR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("All", "REL2C_LEFT", "REL2C_RIGHT", "same_category", "cross_category"):
        value = subsets[name]
        lines.append(
            f"| {name} | {value['N']} | {value['facts_only_correct']}/{value['N']} | "
            f"{value['image_plus_facts_correct']}/{value['N']} | {pct(value['delta_image_minus_facts'])} | "
            f"{value['visual_interference_count']}/{value['visual_interference_denominator']} |"
        )
    lines.extend(
        [
            "",
            "## Error types among facts-correct / image-wrong cases",
            "",
            f"- other_object_color: {error_types.get('other_object_color', 0)}",
            f"- third_color: {error_types.get('third_color', 0)}",
            f"- invalid_or_ambiguous: {error_types.get('invalid_or_ambiguous', 0)}",
            "",
            "## Audit",
            "",
            f"- Results rows: {audit['results_rows']}",
            f"- Case manifest rows: {audit['case_manifest_rows']}",
            f"- Incomplete pairs: {len(audit['incomplete_cases'])}",
            f"- Duplicate records: {len(audit['duplicate_records'])}",
            f"- Ignored result case IDs not in manifest: {len(audit['ignored_case_ids_not_in_manifest'])}",
            "",
        ]
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "pairs": len(paired), "metrics": all_metrics, "mcnemar": mcnemar, "bootstrap": bootstrap}, ensure_ascii=False, indent=2))
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
