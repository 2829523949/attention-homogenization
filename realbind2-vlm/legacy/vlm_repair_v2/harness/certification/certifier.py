"""Counterfactual certification with explicit capability/claim integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .certificate import BindingCertificate, CertificateStatus
from .evidence import EvidenceScore


@dataclass(frozen=True)
class CertificationConfig:
    epsilon_focus: float = -0.05
    tau_remove: float = 0.10
    minimum_base_score: float = 0.40
    calibration_mode: str = "UNCALIBRATED_DIAGNOSTIC"
    # A calibrated claim requires both the mode and an explicit calibration
    # artifact identifier.  The current repository has no such artifact, so
    # smoke runs remain diagnostic.
    calibration_artifact_id: str | None = None

    @property
    def calibrated(self) -> bool:
        return self.calibration_mode == "CALIBRATED_DEVELOPMENT" and bool(self.calibration_artifact_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon_focus": self.epsilon_focus,
            "tau_remove": self.tau_remove,
            "minimum_base_score": self.minimum_base_score,
            "calibration_mode": self.calibration_mode,
            "calibration_artifact_id": self.calibration_artifact_id,
            "threshold_provenance": "hand_set_diagnostic_placeholder_uncalibrated",
        }


def _canonical_claim(value: Any) -> Any:
    if hasattr(value, "bbox_xyxy_pixels"):
        return (
            "region",
            tuple(round(float(item), 8) for item in value.bbox_xyxy_pixels),
            tuple(round(float(item), 8) for item in value.bbox_xyxy_normalized),
        )
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical_claim(item)) for key, item in value.items() if key != "score"))
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_claim(item) for item in value)
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, float):
        return round(value, 8)
    return value


def claim_fingerprint(entity_id: str, predicate: str, proposed_value: Any) -> str:
    payload = {
        "entity_id": entity_id,
        "predicate": predicate,
        "value": _canonical_claim(proposed_value),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


class EvidenceCertifier:
    def __init__(self, config: CertificationConfig | None = None):
        self.config = config or CertificationConfig()

    @staticmethod
    def _certificate_id(entity_id: str, predicate: str, value: Any) -> str:
        raw = f"{entity_id}|{predicate}|{_canonical_claim(value)}".encode("utf-8")
        return "BC_" + hashlib.sha256(raw).hexdigest()[:16]

    def certify(
        self,
        *,
        entity_id: str,
        predicate: str,
        proposed_value: Any,
        binding_kind: str,
        raw: EvidenceScore,
        focus: EvidenceScore,
        remove: EvidenceScore,
        provenance: tuple[str, ...] = (),
        dag_node_id: str | None = None,
        dependency_ids: tuple[str, ...] = (),
        score_contract: Mapping[str, Any] | None = None,
        acquisition_processor: str | None = None,
        view_ids: tuple[str, str, str] | None = ("RAW", "FOCUS", "REMOVE"),
        evidence_kind: str = "ENTITY_BINDING_EVIDENCE",
        identity_layer: str = "FULL_LOGICAL_IDENTITY",
        calibration_status: str | None = None,
        gate_policy: str = "STRICT",
    ) -> BindingCertificate:
        for name, score in (("raw", raw), ("focus", focus), ("remove", remove)):
            if not isinstance(score, EvidenceScore):
                raise TypeError(f"{name} must be EvidenceScore, got {type(score)!r}")

        fingerprint = claim_fingerprint(entity_id, predicate, proposed_value)
        expected_views = tuple(view_ids) if view_ids is not None else ("RAW", "FOCUS", "REMOVE")
        checks: dict[str, bool] = {
            "all_scores_supported": all(score.support == "SUPPORTED" for score in (raw, focus, remove)),
            "all_normalized_scores_present": all(score.normalized_score is not None for score in (raw, focus, remove)),
            "same_processor": len({score.processor for score in (raw, focus, remove)}) == 1,
            "same_normalizer": len({score.normalizer_revision for score in (raw, focus, remove)}) == 1,
            "normalizer_documented": all(score.normalizer_revision not in {"unspecified", "none", ""} for score in (raw, focus, remove)),
            "same_score_kind": len({score.score_kind for score in (raw, focus, remove)}) == 1,
            "fixed_claim_matches": all(_canonical_claim(score.claimed_value) == _canonical_claim(proposed_value) for score in (raw, focus, remove)),
            "views_distinct": len(set(expected_views)) == len(expected_views),
            "view_ids_match": all(score.view_id in (None, view_id) for score, view_id in zip((raw, focus, remove), expected_views)),
        }
        failure_code: str | None = None

        if not checks["all_scores_supported"]:
            failure_code = "CERTIFICATION_SCORE_UNSUPPORTED"
        elif not checks["all_normalized_scores_present"]:
            failure_code = "CERTIFICATION_SCORE_MISSING"
        elif not all(checks[key] for key in ("same_processor", "same_normalizer", "normalizer_documented", "same_score_kind")):
            failure_code = "SCORE_CONTRACT_MISMATCH"
        elif not checks["fixed_claim_matches"]:
            failure_code = "FIXED_CLAIM_MISMATCH"
        elif not checks["views_distinct"] or not checks["view_ids_match"]:
            failure_code = "COUNTERFACTUAL_VIEW_MISMATCH"

        score_contract_summary: dict[str, Any] = {}
        if score_contract is not None:
            score_contract_summary = {
                "processor": score_contract.get("processor"),
                "supports_certification_score": bool(score_contract.get("supports_certification_score")),
                "supports_same_claim_rescoring": bool(score_contract.get("supports_same_claim_rescoring")),
                "comparable_across_counterfactual_views": bool(score_contract.get("comparable_across_counterfactual_views")),
                "score_semantics": score_contract.get("score_semantics"),
                "score_normalization": score_contract.get("score_normalization"),
                "requires_fixed_region": bool(score_contract.get("requires_fixed_region")),
                "requires_fixed_query": bool(score_contract.get("requires_fixed_query")),
                "evidence_kind": score_contract.get("evidence_kind") or evidence_kind,
            }
            if not score_contract_summary["supports_certification_score"]:
                checks["score_contract_supports_certification"] = False
                failure_code = failure_code or "CERTIFICATION_SCORE_UNSUPPORTED"
            else:
                checks["score_contract_supports_certification"] = True
            if not score_contract_summary["supports_same_claim_rescoring"]:
                checks["score_contract_supports_same_claim"] = False
                failure_code = failure_code or "SCORE_CONTRACT_MISMATCH"
            else:
                checks["score_contract_supports_same_claim"] = True
            if not score_contract_summary["comparable_across_counterfactual_views"]:
                checks["score_contract_comparable_views"] = False
                failure_code = failure_code or "SCORE_CONTRACT_MISMATCH"
            else:
                checks["score_contract_comparable_views"] = True
            if score_contract_summary["processor"] and raw.processor != score_contract_summary["processor"]:
                checks["score_processor_matches_contract"] = False
                failure_code = failure_code or "SCORE_CONTRACT_MISMATCH"
            else:
                checks["score_processor_matches_contract"] = True
            declared_normalizer = score_contract_summary.get("score_normalization")
            if declared_normalizer and any(score.normalizer_revision != declared_normalizer for score in (raw, focus, remove)):
                checks["normalizer_matches_contract"] = False
                failure_code = failure_code or "SCORE_CONTRACT_MISMATCH"
            else:
                checks["normalizer_matches_contract"] = True

            if score_contract_summary["requires_fixed_region"]:
                region_ids = [str(score.diagnostics.get("region_id") or "") for score in (raw, focus, remove)]
                crop_policy_ids = [str(score.diagnostics.get("crop_policy_id") or "") for score in (raw, focus, remove)]
                checks["same_fixed_region_id"] = len(set(region_ids)) == 1 and bool(region_ids[0])
                checks["same_crop_policy"] = len(set(crop_policy_ids)) == 1 and bool(crop_policy_ids[0])
                if not checks["same_fixed_region_id"] or not checks["same_crop_policy"]:
                    failure_code = failure_code or "FIXED_REGION_CONTRACT_VIOLATION"
            else:
                checks["same_fixed_region_id"] = True
                checks["same_crop_policy"] = True

            if score_contract_summary["requires_fixed_query"]:
                query_hashes = [str(score.diagnostics.get("query_hash") or "") for score in (raw, focus, remove)]
                template_ids = [str(score.diagnostics.get("query_template_id") or "") for score in (raw, focus, remove)]
                checks["same_query_hash"] = len(set(query_hashes)) == 1 and bool(query_hashes[0])
                checks["same_template_id"] = len(set(template_ids)) == 1 and bool(template_ids[0])
                if not checks["same_query_hash"] or not checks["same_template_id"]:
                    failure_code = failure_code or "FIXED_QUERY_CONTRACT_VIOLATION"
            else:
                checks["same_query_hash"] = True
                checks["same_template_id"] = True

        diagnostics: dict[str, Any] = {
            "claim_fingerprint": fingerprint,
            "checks": checks,
            "failure_code": failure_code,
            "view_ids": list(expected_views),
            "score_processor": raw.processor,
            "acquisition_processor": acquisition_processor,
            "score_contract": score_contract_summary,
            "score_semantics": (score_contract_summary.get("score_semantics") or raw.diagnostics.get("score_semantics") or raw.score_kind),
            "score_normalization": (score_contract_summary.get("score_normalization") or raw.normalizer_revision),
            "evidence_kind": evidence_kind,
            "identity_layer": identity_layer,
            "calibration_status": calibration_status or (
                "CALIBRATED" if self.config.calibrated else self.config.calibration_mode
            ),
            "gate_policy": gate_policy,
        }

        if failure_code is not None:
            rationale = f"certification refused: {failure_code}"
            return BindingCertificate(
                self._certificate_id(entity_id, predicate, proposed_value),
                entity_id,
                predicate,
                proposed_value,
                binding_kind,
                raw,
                focus,
                remove,
                None,
                None,
                CertificateStatus.UNCERTAIN,
                self.config.calibrated,
                self.config.to_dict(),
                provenance,
                rationale,
                diagnostics,
                dag_node_id,
                dependency_ids,
                evidence_kind,
                identity_layer,
                calibration_status or ("CALIBRATED" if self.config.calibrated else self.config.calibration_mode),
            )

        base = float(raw.normalized_score)
        focus_value = float(focus.normalized_score)
        remove_value = float(remove.normalized_score)
        focus_delta = focus_value - base
        necessity_delta = focus_value - remove_value
        pattern_pass = (
            base >= self.config.minimum_base_score
            and focus_delta >= self.config.epsilon_focus
            and necessity_delta >= self.config.tau_remove
        )
        if gate_policy == "NON_CALIBRATED_INFORMATIONAL":
            diagnostics["diagnostic_pattern_pass"] = bool(pattern_pass)
            if not pattern_pass:
                diagnostics["diagnostic_pattern_note"] = (
                    "existing full-binding diagnostic thresholds were not calibrated for this "
                    "semantic-evidence scorer; they are recorded informationally and do not reject the evidence layer"
                )
            status = CertificateStatus.UNCALIBRATED
            rationale = (
                "real fixed-region semantic evidence recorded as UNCALIBRATED diagnostic evidence; "
                "no calibrated threshold exists for this scorer"
            )
        elif not pattern_pass:
            status = CertificateStatus.REJECTED
            rationale = "counterfactual necessity/base gate failed"
        elif self.config.calibrated:
            status = CertificateStatus.CERTIFIED
            rationale = f"counterfactual pattern passed calibrated gates ({self.config.calibration_artifact_id})"
        else:
            status = CertificateStatus.UNCALIBRATED
            rationale = "counterfactual pattern passed diagnostic gates without development calibration artifact"
        return BindingCertificate(
            self._certificate_id(entity_id, predicate, proposed_value),
            entity_id,
            predicate,
            proposed_value,
            binding_kind,
            raw,
            focus,
            remove,
            focus_delta,
            necessity_delta,
            status,
            self.config.calibrated,
            self.config.to_dict(),
            provenance,
            rationale,
            diagnostics,
            dag_node_id,
            dependency_ids,
            evidence_kind,
            identity_layer,
            calibration_status or ("CALIBRATED" if self.config.calibrated else self.config.calibration_mode),
        )
