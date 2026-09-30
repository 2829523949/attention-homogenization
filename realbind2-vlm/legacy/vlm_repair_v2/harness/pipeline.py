from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.processors.base import VisualState
from harness.processors.color_pixel import PixelColorReader
from harness.processors.florence2 import map_generic_regions
from harness.processors.geometry import horizontal_relation, with_geometry


def acquire_visual_state(case: dict[str, Any], localizer: Any, *, pixel_reader: Any | None = None, siglip_reader: Any | None = None, use_siglip_as_primary: bool = False) -> VisualState:
    """LEGACY_GENERIC_PROPOSAL_BASELINE; not the corrected main interface."""
    image_path = Path(case["image_path"])
    generic = localizer.predict(image_path, task="<OD>", fallback_region_proposals=True)
    mapping, mapping_trace = map_generic_regions(generic, {"A": case["object_A"], "B": case["object_B"]})
    query_trace=[]
    # Generic acquisition is always attempted first. Frozen prompts contain
    # useful object nouns, so missing mappings may use Florence's optional
    # query-conditioned task without any localization annotation.
    if hasattr(localizer,"predict_query"):
        for ref in ("A","B"):
            if mapping[ref] is not None: continue
            candidates=localizer.predict_query(image_path,case[f"object_{ref}"])
            mapping[ref]=max(candidates,key=lambda r:((r.score if r.score is not None else float("-inf")),-r.bbox_xyxy_pixels[0])) if candidates else None
            query_trace.append({"operation":"localize_query_fallback","reference_id":ref,"query_category":case[f"object_{ref}"],"candidate_count":len(candidates),"selected":mapping[ref].__dict__ if mapping[ref] else None})
    pixel = pixel_reader or PixelColorReader()
    entities = []
    color_trace = []
    for ref in ("A", "B"):
        region = mapping[ref]
        pixel_result = pixel.read(image_path, region) if region else None
        siglip_result = siglip_reader.read(image_path, region) if (region and siglip_reader is not None) else None
        selected = siglip_result if (use_siglip_as_primary and siglip_result) else pixel_result
        entities.append(with_geometry(ref, region, category=case[f"object_{ref}"], category_confidence=(region.score if region and region.score is not None else 0.), color=selected.color if selected else None, color_confidence=selected.confidence if selected else 0., provenance=tuple(x for x in (region.source if region else None, selected.source if selected else None) if x)))
        color_trace.append({"operation": "read_color", "reference_id": ref, "pixel": pixel_result.__dict__ if pixel_result else None, "siglip": siglip_result.__dict__ if siglip_result else None, "selected": selected.source if selected else None})
    relation_ab = horizontal_relation("A", mapping["A"], "B", mapping["B"])
    relation_ba = horizontal_relation("B", mapping["B"], "A", mapping["A"])
    missing = [ref for ref, region in mapping.items() if region is None]
    return VisualState(case["case_id"], case["qa_type"], tuple(entities), (relation_ab, relation_ba), tuple([{"operation": "localize_generic", "count": len(generic), "regions": [r.__dict__ for r in generic]}] + mapping_trace + query_trace + color_trace), {"missing_entities": missing, "abstained": bool(missing)}, {"florence2": localizer.version, "pixel_color": pixel.version, "siglip": siglip_reader.version if siglip_reader else "not_run", "geometry": "deterministic_center_v1"})
