#!/usr/bin/env python3
"""Paired multi-model Direct vs R4-only analysis on the frozen PACO set.

The point estimate, family-cluster bootstrap, exact McNemar, and Wilson
intervals are imported from the same frozen RealBind2 statistics module used by
the RealBind2 R4 comparison:
    src/repair/strong_integrated/statistics.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import (
    atomic_json, ensure_release_on_path, jsonl, load_paths, sha256_file, stable_hash,
)


def image_cluster_paired(a: list[dict], b: list[dict], draws: int = 10000, seed: int = 20261008):
    """Same image-cluster bootstrap contract as RealBind2 statistics.image_paired."""
    idx = {r["qa_id"]: r for r in b}
    clusters = {}
    for r in a:
        old = idx[r["qa_id"]]
        if r["image_sha256"] != old["image_sha256"]:
            raise ValueError(f"image identity mismatch for {r['qa_id']}")
        key = r["image_sha256"]
        if key not in clusters:
            clusters[key] = [0, 0]
        clusters[key][0] += int(r["correct"]) - int(old["correct"])
        clusters[key][1] += 1
    data = np.array([clusters[k] for k in sorted(clusters)], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        summed = data[rng.integers(len(data), size=len(data))].sum(axis=0)
        values.append(summed[0] / summed[1])
    return {"ci95": np.quantile(values, [0.025, 0.975]).tolist(), "clusters": len(data),
            "seed": seed, "draws": draws}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paths.yaml")
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--model-tags", default="qwen35_4b,qwen3vl_4b,qwen3vl_8b,internvl35_8b")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="defaults to <paths.output_root>/runs")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="defaults to <paths.output_root>/analysis")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    paths = load_paths(args.paths)
    release = ensure_release_on_path(Path(paths["release_root"]))
    from src.repair.strong_integrated.statistics import paired, summarize

    dataset_dir = args.dataset_dir.resolve()
    split_spec = json.loads((dataset_dir / "split_spec.json").read_text())
    task_semantics = split_spec.get("task_semantics", {})
    benchmark = jsonl(dataset_dir / "benchmark.jsonl")
    benchmark_ids = {q["qa_id"] for q in benchmark}
    if len(benchmark_ids) != len(benchmark):
        raise ValueError("duplicate benchmark QA IDs")
    benchmark_sha = sha256_file(dataset_dir / "benchmark.jsonl")
    runs_root = args.runs_root or (Path(paths["output_root"]) / "runs")
    out = args.output_dir or (Path(paths["output_root"]) / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    tags = [x.strip() for x in args.model_tags.split(",") if x.strip()]
    results = []
    diagnostics = []
    missing = []
    for tag in tags:
        direct_dir = runs_root / tag / "direct"
        r4_dir = runs_root / tag / "r4_only"
        if not ((direct_dir / "results.jsonl").is_file() and (r4_dir / "results.jsonl").is_file()):
            missing.append(tag)
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"missing Direct/R4 result pair for {tag}")
        direct = jsonl(direct_dir / "results.jsonl")
        r4 = jsonl(r4_dir / "results.jsonl")
        for label, rows in (("direct", direct), ("r4_only", r4)):
            ids = {r["qa_id"] for r in rows}
            if ids != benchmark_ids or len(rows) != len(benchmark):
                raise ValueError(f"{tag}/{label} does not cover the frozen benchmark exactly")
            if any(r.get("run_benchmark_sha256") != benchmark_sha for r in rows):
                raise ValueError(f"{tag}/{label} benchmark hash mismatch")
            if any(r.get("model_tag") != tag for r in rows):
                raise ValueError(f"{tag}/{label} model tag mismatch")
            if any(r.get("method") != label for r in rows):
                raise ValueError(f"{tag}/{label} method field mismatch")
        if direct[0]["model_id"] != r4[0]["model_id"]:
            raise ValueError(f"{tag}: Direct/R4 model IDs differ")
        direct_identity = json.loads((direct_dir / "identity.json").read_text())
        r4_identity = json.loads((r4_dir / "identity.json").read_text())
        if direct_identity.get("model_config") != r4_identity.get("model_config"):
            raise ValueError(f"{tag}: Direct/R4 model configs differ")
        if direct_identity.get("adapter_metadata", {}).get("model_hash") != r4_identity.get("adapter_metadata", {}).get("model_hash"):
            raise ValueError(f"{tag}: Direct/R4 model metadata differs")

        d_sum = summarize(direct)
        r_sum = summarize(r4)
        pair = paired(r4, direct)
        image_pair = image_cluster_paired(r4, direct)
        by_family = {}
        for q in benchmark:
            by_family.setdefault(q["family_id"], []).append(q["qa_id"])
        selector_fallback_tasks = set()
        selector_no_candidate_tasks = set()
        target_attr_valid = 0
        target_attr_correct = 0
        clean_correct = 0
        for row in r4:
            stages = row.get("stages") or {}
            sel = stages.get("selection") or {}
            for ref in ("A", "B"):
                task_id = f"{row['family_id']}|{ref}"
                s = sel.get(task_id) or {}
                if s.get("qwen_status") == "QWEN_SELECTOR_INVALID_FALLBACK_NATIVE_TOP1":
                    selector_fallback_tasks.add(task_id)
                if s.get("status") == "NO_CANDIDATE":
                    selector_no_candidate_tasks.add(task_id)
            target_mask = stages.get("target_mask")
            if target_mask and target_mask.get("attribute_valid"):
                target_attr_valid += 1
            if row.get("attribute_correct"):
                target_attr_correct += 1
            if row.get("clean_correct"):
                clean_correct += 1
        selector_fallback = len(selector_fallback_tasks)
        selector_no_candidate = len(selector_no_candidate_tasks)
        record = {
            "model_tag": tag,
            "model_id": direct[0]["model_id"],
            "n": len(benchmark),
            "families": len(by_family),
            "unique_images": len({q["image_sha256"] for q in benchmark}),
            "benchmark_sha256": benchmark_sha,
            "direct": d_sum,
            "r4_only": r_sum,
            "paired_r4_minus_direct": pair,
            "paired_image_bootstrap": image_pair,
            "net_correct_answers": pair["discordant_wins"] - pair["discordant_losses"],
            "selector_invalid_fallback_count": selector_fallback,
            "selector_no_candidate_count": selector_no_candidate,
            "target_attribute_valid": target_attr_valid,
            "target_attribute_correct": target_attr_correct,
            "clean_correct": clean_correct,
            "direct_identity": direct_identity,
            "r4_identity": r4_identity,
            "release_root": str(release),
        }
        results.append(record)
        for q, d_row, r_row in zip(benchmark, direct, r4):
            if q["qa_id"] != d_row["qa_id"] or q["qa_id"] != r_row["qa_id"]:
                raise ValueError("row ordering mismatch")
            diagnostics.append({
                "model_tag": tag,
                "model_id": direct[0]["model_id"],
                "qa_id": q["qa_id"],
                "family_id": q["family_id"],
                "paco_split": q["paco_split"],
                "qa_type": q["qa_type"],
                "direct_correct": d_row["correct"],
                "direct_valid": d_row["valid"],
                "r4_correct": r_row["correct"],
                "r4_valid": r_row["valid"],
                "rescue": r_row["correct"] and not d_row["correct"],
                "regression": not r_row["correct"] and d_row["correct"],
                "logical_target_correct": r_row.get("logical_target_correct"),
                "r4_target_grounded": r_row.get("r4_target_grounded"),
                "attribute_correct": r_row.get("attribute_correct"),
                "geometry_valid": r_row.get("geometry_valid"),
                "minimal_state_sufficient": r_row.get("minimal_state_sufficient"),
                "clean_correct": r_row.get("clean_correct"),
            })

    analysis = {
        "schema": "paco-realbind2-direct-vs-r4-v1",
        "scope": "conditioned PACO-LVIS; retrospectively method-conditioned by primary R4 selector",
        "dataset_dir": str(dataset_dir),
        "benchmark_sha256": benchmark_sha,
        "task_semantics": task_semantics,
        "models": results,
        "missing_models": missing,
        "statistics_source": "frozen RealBind2 src/repair/strong_integrated/statistics.py",
        "statistics_version": "repair-statistics-family-cluster-v1",
    }
    atomic_json(out / "analysis.json", analysis)
    # Full diagnostics
    with (out / "paired_diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
        w.writeheader(); w.writerows(diagnostics)
    # Paper table
    lines = [
        "# PACO: frozen conditioned Direct vs R4-only",
        "",
        "Conditioning: ref_A != ref_B and R4-predicted A/B IoU < 0.1, selected by the primary Qwen3.5-4B R4 selector.",
        f"Color semantics in this benchmark: {task_semantics.get('question_color_semantics', 'unknown')}.",
        "All models are evaluated on exactly the same conditioned QA set. Intervals are family-cluster bootstrap 95% CIs.",
        "",
        "| Model | Direct Acc | R4-only Acc | Gain pp [95% CI] | Rescue | Regression | Net | Selector fallback | Invalid Direct/R4 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        p = r["paired_r4_minus_direct"]
        ci = "[" + ", ".join(f"{100 * v:.2f}" for v in p["ci95_family_bootstrap"]) + "]"
        lines.append(
            "| " + " | ".join([
                r["model_id"],
                f"{100 * r['direct']['overall']['accuracy']:.2f}%",
                f"{100 * r['r4_only']['overall']['accuracy']:.2f}%",
                f"{100 * p['paired_accuracy_difference']:.2f} {ci}",
                str(p["discordant_wins"]),
                str(p["discordant_losses"]),
                str(r["net_correct_answers"]),
                str(r["selector_invalid_fallback_count"] + r["selector_no_candidate_count"]),
                f"{r['direct']['overall']['invalid']}/{r['r4_only']['overall']['invalid']}",
            ]) + " |"
        )
    (out / "paper_table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
