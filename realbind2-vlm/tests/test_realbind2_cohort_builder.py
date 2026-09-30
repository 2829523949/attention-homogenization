from scripts.build_realbind2_cohorts import matched_f0, stable_f1_rows


def test_stable_f1_requires_independent_agreement() -> None:
    case = {
        "case_id": "F__REL2C_LEFT",
        "qa_id": "F__REL2C_LEFT",
        "prerequisite_qa_ids": ["F__I2C_A", "F__I2C_B"],
    }
    baseline = {
        key: {"prediction_normalized": value, "correct": correct, "valid": True}
        for key, value, correct in (
            ("F__I2C_A", "red", True),
            ("F__I2C_B", "blue", True),
            ("F__REL2C_LEFT", "blue", False),
        )
    }
    assert len(stable_f1_rows([case], baseline, {key: dict(row) for key, row in baseline.items()})) == 1
    changed = {key: dict(row) for key, row in baseline.items()}
    changed["F__REL2C_LEFT"]["prediction_normalized"] = "green"
    assert stable_f1_rows([case], baseline, changed) == []


def test_matching_is_without_replacement() -> None:
    common = {
        "relation_direction": "LEFT",
        "target_role": "A",
        "target_attribute": "red",
        "target_category": "car",
        "target_bbox_scale": "medium",
        "target_side": "LEFT",
        "target_bbox_area_fraction": 0.1,
    }
    stable = [{**common, "case_id": "f1a"}, {**common, "case_id": "f1b"}]
    f0 = [{**common, "case_id": "f0a"}, {**common, "case_id": "f0b"}]
    result = matched_f0(stable, f0)
    assert len({row["case_id"] for row in result}) == 2
    assert {row["matched_f1_case_id"] for row in result} == {"f1a", "f1b"}

