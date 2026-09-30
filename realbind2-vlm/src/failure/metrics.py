from __future__ import annotations

from collections import Counter, defaultdict

COMPOSITION_TYPES = {"REL2C_LEFT", "REL2C_RIGHT"}


def baseline_metrics(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row["qa_type"])].append(row)
    atom_types = {"I2C_A", "I2C_B"}
    relation_types = {"PAIR_REL_A_TO_B", "REL2I_LEFT", "REL2I_RIGHT"}

    def aggregate(selected: list[dict]) -> dict:
        n = len(selected)
        correct = sum(bool(row.get("correct")) for row in selected)
        invalid = sum(not bool(row.get("valid", not row.get("invalid_prediction", False))) for row in selected)
        return {"n": n, "correct": correct, "accuracy": correct / n if n else None, "invalid": invalid, "invalid_rate": invalid / n if n else None}

    all_rows = list(rows)
    atomic = [row for row in rows if row["qa_type"] in atom_types]
    relation = [row for row in rows if row["qa_type"] in relation_types]
    composition = [row for row in rows if row["qa_type"] in COMPOSITION_TYPES]
    indexed: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        indexed[str(row["family_id"])][str(row["qa_type"])] = row
    atomic_ready = atomic_fail = binding_ready = binding_fail = 0
    for family in indexed.values():
        atom_names = ("I2C_A", "I2C_B", "PAIR_REL_A_TO_B")
        if any(name not in family for name in atom_names):
            continue
        atoms_ok = all(bool(family[name].get("correct")) for name in atom_names)
        for side in ("LEFT", "RIGHT"):
            comp_name, binding_name = f"REL2C_{side}", f"REL2I_{side}"
            if comp_name not in family:
                continue
            if atoms_ok:
                atomic_ready += 1
                atomic_fail += not bool(family[comp_name].get("correct"))
            if atoms_ok and binding_name in family and bool(family[binding_name].get("correct")):
                binding_ready += 1
                binding_fail += not bool(family[comp_name].get("correct"))
    return {
        "overall": aggregate(all_rows),
        "atomic": aggregate(atomic),
        "relation": aggregate(relation),
        "composition": aggregate(composition),
        "PCFR_atomic": atomic_fail / atomic_ready if atomic_ready else None,
        "PCFR_atomic_n": atomic_ready,
        "PCFR_binding": binding_fail / binding_ready if binding_ready else None,
        "PCFR_binding_n": binding_ready,
        "by_type": {key: aggregate(value) for key, value in sorted(by_type.items())},
    }


def visual_interference_metrics(rows: list[dict]) -> dict:
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["condition"])] = row
    complete = [conditions for conditions in by_case.values() if {"original", "facts_only", "image_correct_facts", "defocus_correct_facts"}.issubset(conditions)]
    facts_correct = sum(bool(c["facts_only"]["correct"]) for c in complete)
    interference = sum(bool(c["facts_only"]["correct"]) and not bool(c["image_correct_facts"]["correct"]) for c in complete)
    rescue = sum(not bool(c["original"]["correct"]) and bool(c["facts_only"]["correct"]) for c in complete)
    regression = sum(bool(c["original"]["correct"]) and not bool(c["image_correct_facts"]["correct"]) for c in complete)
    error_types = Counter(str(row.get("error_type", "none")) for row in rows)
    return {
        "n_complete_cases": len(complete),
        "VIR": interference / facts_correct if facts_correct else None,
        "fact_rescue": rescue / len(complete) if complete else None,
        "image_induced_regression": regression / len(complete) if complete else None,
        "error_type_counts": dict(sorted(error_types.items())),
    }


def classify_composition_error(prediction: str | None, target_color: str, competitor_color: str) -> str:
    if prediction is None:
        return "invalid_error"
    normalized = prediction.casefold()
    if normalized == target_color.casefold():
        return "correct"
    if normalized == competitor_color.casefold():
        return "competitor_attribute_error"
    return "third_value_error"
