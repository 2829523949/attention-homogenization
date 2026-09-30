from __future__ import annotations

QA_ORDER = ("I2C_A", "I2C_B", "PAIR_REL_A_TO_B", "REL2I_LEFT", "REL2I_RIGHT", "REL2C_LEFT", "REL2C_RIGHT")
TEMPLATES = {
    "I2C_A": "What color is {A}? Answer with one color.",
    "I2C_B": "What color is {B}? Answer with one color.",
    "PAIR_REL_A_TO_B": "Is {A} to the LEFT or RIGHT of {B}? Answer LEFT or RIGHT.",
    "REL2I_LEFT": "Between {A} and {B}, which one is on the LEFT? Answer A or B.",
    "REL2I_RIGHT": "Between {A} and {B}, which one is on the RIGHT? Answer A or B.",
    "REL2C_LEFT": "Between {A} and {B}, what color is the one on the LEFT? Answer with one color.",
    "REL2C_RIGHT": "Between {A} and {B}, what color is the one on the RIGHT? Answer with one color.",
}


def generate_questions(family: dict) -> list[dict]:
    left = "A" if family["geometry_metrics"]["left_is_A"] else "B"
    right = "B" if left == "A" else "A"
    colors = {"A": family["color_A"], "B": family["color_B"]}
    identity = {"A": "A", "B": "B"}
    answers = {
        "I2C_A": colors["A"], "I2C_B": colors["B"],
        "PAIR_REL_A_TO_B": "LEFT" if left == "A" else "RIGHT",
        "REL2I_LEFT": identity[left], "REL2I_RIGHT": identity[right],
        "REL2C_LEFT": colors[left], "REL2C_RIGHT": colors[right],
    }
    out = []
    for qa_type in QA_ORDER:
        out.append({"family_id": family["family_id"], "qa_id": f"{family['family_id']}__{qa_type}", "qa_type": qa_type,
                    "question": TEMPLATES[qa_type].format(A=family["referent_A"], B=family["referent_B"]),
                    "answer_type": "position" if qa_type == "PAIR_REL_A_TO_B" else "identity" if qa_type.startswith("REL2I") else "color",
                    "ground_truth": answers[qa_type]})
    return out


DEPENDENCIES = {
    "atomic_prerequisites": ["I2C_A", "I2C_B", "PAIR_REL_A_TO_B"],
    "relational_identity_readout": {"REL2I_LEFT": ["PAIR_REL_A_TO_B"], "REL2I_RIGHT": ["PAIR_REL_A_TO_B"]},
    "binding_prerequisites": {"REL2C_LEFT": ["I2C_A", "I2C_B", "PAIR_REL_A_TO_B", "REL2I_LEFT"],
                              "REL2C_RIGHT": ["I2C_A", "I2C_B", "PAIR_REL_A_TO_B", "REL2I_RIGHT"]},
    "metrics": {"PCFR_atomic": "P(REL2C fail | I2C_A correct AND I2C_B correct AND PAIR_REL_A_TO_B correct)",
                "PCFR_binding": "P(REL2C side fail | atomic prerequisites correct AND corresponding REL2I correct)"}
}
