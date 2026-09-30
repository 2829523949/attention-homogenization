from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_TOOLS = frozenset({"localize_generic", "localize_query", "refine_mask", "read_color", "read_category", "compute_relation"})


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if self.tool not in ALLOWED_TOOLS: raise ValueError(f"tool not allowlisted: {self.tool}")


@dataclass(frozen=True)
class Plan:
    schema_version: str
    case_id: str
    task: str
    calls: tuple[ToolCall, ...]

    def to_dict(self) -> dict[str, Any]: return asdict(self)


class DeterministicPlanner:
    version = "realbind_task_semantics_v1"

    def plan(self, case: dict[str, Any]) -> Plan:
        task = str(case["qa_type"])
        if task.startswith("REL2C_"):
            calls = (
                ToolCall("localize_generic", {"image_path": case["image_path"]}),
                ToolCall("read_color", {"entity": "A"}), ToolCall("read_color", {"entity": "B"}),
                ToolCall("compute_relation", {"source": "A", "target": "B"}),
            )
        elif task.startswith("REL2I_") or task == "PAIR_REL_A_TO_B":
            calls = (ToolCall("localize_generic", {"image_path": case["image_path"]}), ToolCall("compute_relation", {"source": "A", "target": "B"}))
        elif task in {"I2C_A", "I2C_B"}:
            entity = task[-1]
            calls = (ToolCall("localize_generic", {"image_path": case["image_path"]}), ToolCall("read_color", {"entity": entity}))
        else: raise ValueError(f"unsupported task: {task}")
        return Plan("realbind_strict_tool_plan_v1", str(case["qa_id"]), task, calls)


def parse_llm_plan(payload: dict[str, Any]) -> Plan:
    """Future planner boundary: strict keys and an allowlisted tool vocabulary."""
    if set(payload) != {"schema_version", "case_id", "task", "calls"}: raise ValueError("planner JSON has unknown/missing keys")
    calls = []
    for raw in payload["calls"]:
        if set(raw) != {"tool", "arguments"} or not isinstance(raw["arguments"], dict): raise ValueError("invalid tool call schema")
        calls.append(ToolCall(str(raw["tool"]), dict(raw["arguments"])))
    return Plan(str(payload["schema_version"]), str(payload["case_id"]), str(payload["task"]), tuple(calls))
