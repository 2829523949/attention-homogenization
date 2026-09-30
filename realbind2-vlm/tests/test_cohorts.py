from src.failure.build_cohorts import derive_cohorts
from src.failure.metrics import baseline_metrics


def row(family, qa_type, correct):
    return {"family_id": family, "qa_type": qa_type, "qa_id": f"{family}__{qa_type}", "correct": correct}


def test_f1_requires_every_declared_prerequisite():
    rows = [
        row("f", "I2C_A", True), row("f", "I2C_B", True), row("f", "PAIR_REL_A_TO_B", True),
        row("f", "REL2I_LEFT", True), row("f", "REL2I_RIGHT", True),
        row("f", "REL2C_LEFT", False), row("f", "REL2C_RIGHT", True),
    ]
    f0, f1 = derive_cohorts(rows)
    assert [entry["qa_type"] for entry in f1] == ["REL2C_LEFT"]
    assert [entry["qa_type"] for entry in f0] == ["REL2C_RIGHT"]
    rows[0]["correct"] = False
    f0, f1 = derive_cohorts(rows)
    assert not f0 and not f1


def test_pcfr_uses_same_dependency_gate():
    rows = [
        row("f", "I2C_A", True), row("f", "I2C_B", True), row("f", "PAIR_REL_A_TO_B", True),
        row("f", "REL2I_LEFT", True), row("f", "REL2I_RIGHT", False),
        row("f", "REL2C_LEFT", False), row("f", "REL2C_RIGHT", False),
    ]
    for item in rows:
        item["valid"] = True
    metrics = baseline_metrics(rows)
    assert metrics["PCFR_atomic"] == 1.0 and metrics["PCFR_atomic_n"] == 2
    assert metrics["PCFR_binding"] == 1.0 and metrics["PCFR_binding_n"] == 1
