"""Explicit visual/reasoning obligations; the planner never predicts an answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from harness.referents.referent_resolver import ResolvedReferents


class Operation(str, Enum):
    RESOLVE_REFERENT = "RESOLVE_REFERENT"
    LOCALIZE = "LOCALIZE"
    READ_CATEGORY = "READ_CATEGORY"
    READ_COLOR = "READ_COLOR"
    READ_STATE = "READ_STATE"
    GET_POSITION = "GET_POSITION"
    COMPARE_HORIZONTAL = "COMPARE_HORIZONTAL"
    SELECT_RELATION = "SELECT_RELATION"
    CERTIFY_BINDING = "CERTIFY_BINDING"
    COMPOSE = "COMPOSE"
    OUTPUT = "OUTPUT"


@dataclass(frozen=True)
class PredicateNode:
    node_id: str
    operation: Operation
    required_inputs: tuple[str, ...] = ()
    produced_variable: str | None = None
    requires_visual_acquisition: bool = False
    deterministic: bool = False
    parent_dependencies: tuple[str, ...] = ()
    predicate: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "operation": self.operation.value,
            "required_inputs": list(self.required_inputs),
            "produced_variable": self.produced_variable,
            "requires_visual_acquisition": self.requires_visual_acquisition,
            "deterministic": self.deterministic,
            "parent_dependencies": list(self.parent_dependencies),
            "predicate": self.predicate,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PredicateDAG:
    schema_version: str
    task_type: str
    question: str
    nodes: tuple[PredicateNode, ...]
    edges: tuple[tuple[str, str], ...]
    answer_operation: str

    def __post_init__(self) -> None:
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("PredicateDAG node IDs must be unique")
        node_set = set(ids)
        for source, target in self.edges:
            if source not in node_set or target not in node_set:
                raise ValueError(f"DAG edge references unknown node: {source}->{target}")
        self.topological_order()

    def topological_order(self) -> tuple[PredicateNode, ...]:
        by_id = {node.node_id: node for node in self.nodes}
        incoming = {node.node_id: 0 for node in self.nodes}
        outgoing: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for source, target in self.edges:
            incoming[target] += 1
            outgoing[source].append(target)
        ready = sorted(node_id for node_id, degree in incoming.items() if degree == 0)
        ordered: list[str] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for target in sorted(outgoing[current]):
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(ordered) != len(self.nodes):
            raise ValueError("PredicateDAG contains a cycle")
        return tuple(by_id[node_id] for node_id in ordered)

    def required_visual_nodes(self) -> tuple[PredicateNode, ...]:
        return tuple(node for node in self.topological_order() if node.requires_visual_acquisition)

    def node(self, node_id: str) -> PredicateNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(f"unknown DAG node: {node_id}")

    def output_node(self) -> PredicateNode:
        matches = [node for node in self.nodes if node.operation is Operation.OUTPUT]
        if len(matches) != 1:
            raise ValueError("PredicateDAG must have exactly one OUTPUT node")
        return matches[0]

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        by_target: dict[str, list[str]] = {}
        for source, target in self.edges:
            by_target.setdefault(target, []).append(source)
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            for parent in by_target.get(current, []):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return tuple(sorted(seen))

    def required_node_ids(self) -> tuple[str, ...]:
        """Nodes on a path to OUTPUT, excluding OUTPUT itself."""
        output = self.output_node().node_id
        return tuple(node_id for node_id in self.topological_order_ids() if node_id in set(self.ancestors(output)))

    def topological_order_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.topological_order())

    def required_proof_nodes(self) -> tuple[PredicateNode, ...]:
        """DAG obligations that must be evidenced before language reasoning.

        RESOLVE_REFERENT is a parser obligation; COMPOSE/OUTPUT are language
        operations.  Everything else on a path to OUTPUT (localization,
        binding certification, attribute acquisition/certification, positions,
        comparisons, and selection) is required.
        """

        proof_operations = {
            Operation.LOCALIZE,
            Operation.READ_CATEGORY,
            Operation.READ_COLOR,
            Operation.READ_STATE,
            Operation.GET_POSITION,
            Operation.COMPARE_HORIZONTAL,
            Operation.SELECT_RELATION,
            Operation.CERTIFY_BINDING,
        }
        required_ids = set(self.required_node_ids())
        return tuple(node for node in self.topological_order() if node.node_id in required_ids and node.operation in proof_operations)

    def required_logical_refs(self) -> tuple[str, ...]:
        refs: set[str] = set()
        for node in self.required_proof_nodes():
            for value in node.required_inputs:
                if value in {"A", "B", "TARGET"}:
                    refs.add(value)
                elif isinstance(value, str) and value.startswith("E_") and value[2:] in {"A", "B", "TARGET"}:
                    refs.add(value[2:])
        return tuple(sorted(refs))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "question": self.question,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
            "answer_operation": self.answer_operation,
        }


def _node(
    node_id: str,
    operation: Operation,
    *,
    required_inputs: Iterable[str] = (),
    produced_variable: str | None = None,
    requires_visual_acquisition: bool = False,
    deterministic: bool = False,
    parents: Iterable[str] = (),
    predicate: str | None = None,
    notes: str = "",
) -> PredicateNode:
    return PredicateNode(
        node_id,
        operation,
        tuple(required_inputs),
        produced_variable,
        requires_visual_acquisition,
        deterministic,
        tuple(parents),
        predicate,
        notes,
    )


def build_predicate_dag(question: str, task_type: str, referents: ResolvedReferents) -> PredicateDAG:
    """Build obligations for the audited task types.

    The DAG contains no answer, color label, bbox, or evaluator field. A
    resolver can therefore fail cleanly when the legal interface is absent.
    """

    if referents.status.value == "REFERENT_INTERFACE_UNDERSPECIFIED":
        raise ValueError("cannot build an automatic PredicateDAG for an underspecified referent interface")
    specs = {spec.logical_ref: spec for spec in referents.specs}
    refs = tuple(specs)
    nodes: list[PredicateNode] = []
    edges: list[tuple[str, str]] = []
    for logical_ref in refs:
        resolve_id = f"resolve_{logical_ref}"
        localize_id = f"localize_{logical_ref}"
        certify_id = f"certify_region_{logical_ref}"
        nodes.extend(
            (
                _node(resolve_id, Operation.RESOLVE_REFERENT, required_inputs=(logical_ref,), produced_variable=logical_ref, notes="bind visible semantic description to a logical reference"),
                _node(localize_id, Operation.LOCALIZE, required_inputs=(logical_ref,), produced_variable=f"E_{logical_ref}", requires_visual_acquisition=True, parents=(resolve_id,), predicate="REFERENT_TO_REGION"),
                _node(certify_id, Operation.CERTIFY_BINDING, required_inputs=(f"E_{logical_ref}",), produced_variable=f"cert_region_{logical_ref}", requires_visual_acquisition=True, parents=(localize_id,), predicate="REFERENT_TO_REGION"),
            )
        )
        edges.extend(((resolve_id, localize_id), (localize_id, certify_id)))

    if task_type in {"REL2C_LEFT", "REL2C_RIGHT", "REL2I_LEFT", "REL2I_RIGHT", "PAIR_REL_A_TO_B"} and {"A", "B"}.issubset(specs):
        position_a = _node("position_A", Operation.GET_POSITION, required_inputs=("E_A",), produced_variable="position_A", deterministic=True, parents=("certify_region_A",), predicate="CENTER")
        position_b = _node("position_B", Operation.GET_POSITION, required_inputs=("E_B",), produced_variable="position_B", deterministic=True, parents=("certify_region_B",), predicate="CENTER")
        compare = _node("compare_horizontal", Operation.COMPARE_HORIZONTAL, required_inputs=("position_A", "position_B"), produced_variable="relation_A_B", deterministic=True, parents=("position_A", "position_B"), predicate="HORIZONTAL_RELATION")
        nodes.extend((position_a, position_b, compare))
        edges.extend((("certify_region_A", "position_A"), ("certify_region_B", "position_B"), ("position_A", "compare_horizontal"), ("position_B", "compare_horizontal")))
        if task_type in {"REL2C_LEFT", "REL2C_RIGHT", "REL2I_LEFT", "REL2I_RIGHT"}:
            side = "left" if task_type.endswith("LEFT") else "right"
            select = _node("select_target", Operation.SELECT_RELATION, required_inputs=("relation_A_B", side), produced_variable="selected_entity", deterministic=True, parents=("compare_horizontal",), predicate="SELECT_RELATION", notes=f"select the {side} entity")
            nodes.append(select)
            edges.append(("compare_horizontal", "select_target"))
            if task_type.startswith("REL2C"):
                read = _node("read_selected_color", Operation.READ_COLOR, required_inputs=("selected_entity",), produced_variable="selected_color", requires_visual_acquisition=True, parents=("select_target",), predicate="COLOR")
                certify_color = _node("certify_selected_color", Operation.CERTIFY_BINDING, required_inputs=("selected_entity", "selected_color"), produced_variable="cert_selected_color", requires_visual_acquisition=True, parents=("read_selected_color",), predicate="ENTITY_TO_COLOR")
                nodes.extend((read, certify_color))
                edges.extend((("select_target", "read_selected_color"), ("read_selected_color", "certify_selected_color")))
                compose_parent = "certify_selected_color"
            else:
                compose_parent = "select_target"
        else:
            compose_parent = "compare_horizontal"
        compose = _node("compose", Operation.COMPOSE, required_inputs=("question", "selected_entity" if task_type != "PAIR_REL_A_TO_B" else "relation_A_B"), produced_variable="compositional_result", parents=(compose_parent,), notes="language-only composition after visual obligations")
        output = _node("output", Operation.OUTPUT, required_inputs=("compositional_result",), produced_variable="answer", parents=("compose",), notes="final answer is created only by the reasoner")
        nodes.extend((compose, output))
        edges.extend(((compose_parent, "compose"), ("compose", "output")))
        answer_operation = "language_reasoning"
    elif task_type in {"I2C_A", "I2C_B"} and len(refs) == 1:
        logical_ref = refs[0]
        read = _node("read_color", Operation.READ_COLOR, required_inputs=(f"E_{logical_ref}",), produced_variable="color", requires_visual_acquisition=True, parents=(f"certify_region_{logical_ref}",), predicate="COLOR")
        certify = _node("certify_color", Operation.CERTIFY_BINDING, required_inputs=(f"E_{logical_ref}", "color"), produced_variable="cert_color", requires_visual_acquisition=True, parents=("read_color",), predicate="ENTITY_TO_COLOR")
        compose = _node("compose", Operation.COMPOSE, required_inputs=("question", "color"), produced_variable="compositional_result", parents=("certify_color",))
        output = _node("output", Operation.OUTPUT, required_inputs=("compositional_result",), parents=("compose",), produced_variable="answer")
        nodes.extend((read, certify, compose, output))
        edges.extend(((f"certify_region_{logical_ref}", "read_color"), ("read_color", "certify_color"), ("certify_color", "compose"), ("compose", "output")))
        answer_operation = "language_reasoning"
    else:
        raise ValueError(f"unsupported audited task type/referents: {task_type}/{refs}")

    return PredicateDAG("realbind_corrected_predicate_dag_v1", task_type, question, tuple(nodes), tuple(edges), answer_operation)

