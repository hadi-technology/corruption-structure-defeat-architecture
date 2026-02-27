from __future__ import annotations

from typing import Dict, List


def classify_schema(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    reach = int(out.get("reach_depth", 0))

    if reach <= 0:
        out["classification"] = "terminal"
        out["classification_reason"] = "No downstream schema can consume this schema's output"
    else:
        out["classification"] = "cascade-eligible"
        out["classification_reason"] = "At least one downstream schema can consume this output"

    if reach >= 2:
        out["cascade_tier"] = "multi-hop-cascade"
    elif reach == 1:
        out["cascade_tier"] = "single-hop-cascade"
    else:
        out["cascade_tier"] = "terminal"

    return out


def classify_all_schemas(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [classify_schema(r) for r in rows]
