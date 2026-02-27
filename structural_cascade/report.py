from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _fraction(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _family_depth_breakdown(
    rows: List[Dict[str, object]],
    key: str,
) -> Dict[str, Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        bucket = str(row.get(key, "UNKNOWN"))
        buckets.setdefault(bucket, []).append(row)

    out: Dict[str, Dict[str, object]] = {}
    for bucket, items in sorted(buckets.items()):
        total = len(items)
        terminal = sum(1 for r in items if str(r.get("classification", "")) == "terminal")
        cascade_eligible = sum(1 for r in items if str(r.get("classification", "")) == "cascade-eligible")
        multi_hop = sum(1 for r in items if int(r.get("reach_depth", 0)) >= 2)
        out[bucket] = {
            "total": total,
            "terminal": terminal,
            "cascade_eligible": cascade_eligible,
            "multi_hop": multi_hop,
            "terminal_pct": _fraction(terminal, total),
            "cascade_eligible_pct": _fraction(cascade_eligible, total),
            "multi_hop_pct": _fraction(multi_hop, total),
            "avg_reach_depth": _mean([float(r.get("reach_depth", 0)) for r in items]),
            "avg_reachable_schema_count": _mean(
                [float(r.get("reachable_schema_count", 0)) for r in items]
            ),
        }
    return out


def _simulation_summary(simulation_rows: List[Dict[str, object]]) -> Dict[str, object]:
    statuses: Dict[str, int] = {}
    classes: Dict[str, int] = {}

    for row in simulation_rows:
        status = str(row.get("simulation_simulation_status", "missing"))
        statuses[status] = statuses.get(status, 0) + 1
        sim_cls = str(row.get("simulation_simulation_classification", "unknown"))
        classes[sim_cls] = classes.get(sim_cls, 0) + 1

    ok_rows = [r for r in simulation_rows if str(r.get("simulation_simulation_status", "")) == "ok"]
    consumed = sum(1 for r in ok_rows if bool(r.get("simulation_corrupted_consequent_consumed", False)))
    changed = sum(1 for r in ok_rows if bool(r.get("simulation_answer_changed", False)))
    harmful = sum(1 for r in ok_rows if str(r.get("simulation_simulation_classification", "")) == "harmful-cascade")

    return {
        "simulation_count": len(simulation_rows),
        "simulation_ok_count": len(ok_rows),
        "simulation_status_counts": statuses,
        "simulation_class_counts": classes,
        "simulation_consumed_count": consumed,
        "simulation_consumed_pct_of_ok": _fraction(consumed, len(ok_rows)),
        "simulation_answer_changed_count": changed,
        "simulation_answer_changed_pct_of_ok": _fraction(changed, len(ok_rows)),
        "simulation_harmful_count": harmful,
        "simulation_harmful_pct_of_ok": _fraction(harmful, len(ok_rows)),
    }


def generate_report(
    classified_rows: List[Dict[str, object]],
    problem_summaries: List[Dict[str, object]],
    simulation_rows: List[Dict[str, object]] | None = None,
) -> Dict[str, object]:
    total = len(classified_rows)
    terminal = sum(1 for r in classified_rows if str(r.get("classification", "")) == "terminal")
    cascade_eligible = sum(
        1 for r in classified_rows if str(r.get("classification", "")) == "cascade-eligible"
    )
    multi_hop = sum(1 for r in classified_rows if int(r.get("reach_depth", 0)) >= 2)
    avg_reach_depth = _mean([float(r.get("reach_depth", 0)) for r in classified_rows])
    avg_reachable = _mean([float(r.get("reachable_schema_count", 0)) for r in classified_rows])

    p_total = len(problem_summaries)
    p_redirect_any = sum(
        1 for p in problem_summaries if float(p.get("redirect_exposure_ratio", 0.0)) > 0.0
    )
    p_multihop_any = sum(
        1 for p in problem_summaries if float(p.get("multi_hop_exposure_ratio", 0.0)) > 0.0
    )

    report: Dict[str, object] = {
        "total_schemas_analyzed": total,
        "terminal_count": terminal,
        "terminal_pct": _fraction(terminal, total),
        "cascade_eligible_count": cascade_eligible,
        "cascade_eligible_pct": _fraction(cascade_eligible, total),
        "multi_hop_count": multi_hop,
        "multi_hop_pct": _fraction(multi_hop, total),
        "avg_reach_depth": avg_reach_depth,
        "avg_reachable_schema_count": avg_reachable,
        "problems_analyzed": p_total,
        "problem_redirect_exposure_mean": _mean(
            [float(p.get("redirect_exposure_ratio", 0.0)) for p in problem_summaries]
        ),
        "problem_multi_hop_exposure_mean": _mean(
            [float(p.get("multi_hop_exposure_ratio", 0.0)) for p in problem_summaries]
        ),
        "problems_with_redirect_exposure_count": p_redirect_any,
        "problems_with_redirect_exposure_pct": _fraction(p_redirect_any, p_total),
        "problems_with_multi_hop_exposure_count": p_multihop_any,
        "problems_with_multi_hop_exposure_pct": _fraction(p_multihop_any, p_total),
        "by_family": _family_depth_breakdown(classified_rows, "schema_family"),
        "by_problem_depth": _family_depth_breakdown(classified_rows, "problem_depth"),
    }

    sim_summary = _simulation_summary(simulation_rows or [])
    report.update(sim_summary)
    return report


def _render_breakdown_table(title: str, rows: Dict[str, Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    lines.append(f"### {title}")
    lines.append("| Bucket | Total | Terminal | Cascade-eligible | Multi-hop | Avg reach depth |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for bucket in sorted(rows.keys()):
        row = rows[bucket]
        lines.append(
            f"| {bucket} | {int(row.get('total', 0))} | {int(row.get('terminal', 0))} "
            f"| {int(row.get('cascade_eligible', 0))} | {int(row.get('multi_hop', 0))} "
            f"| {float(row.get('avg_reach_depth', 0.0)):.3f} |"
        )
    lines.append("")
    return lines


def build_markdown_report(report: Dict[str, object], dataset_key: str) -> str:
    lines: List[str] = []
    lines.append("# Structural Cascade Report")
    lines.append("")
    lines.append("- Method: `structural cascade potential analysis`")
    lines.append(f"- Dataset: `{dataset_key}`")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- Total schemas analyzed: `{int(report.get('total_schemas_analyzed', 0))}`")
    lines.append(f"- Terminal schemas: `{int(report.get('terminal_count', 0))}` ({100.0 * float(report.get('terminal_pct', 0.0)):.2f}%)")
    lines.append(
        f"- Cascade-eligible schemas: `{int(report.get('cascade_eligible_count', 0))}` "
        f"({100.0 * float(report.get('cascade_eligible_pct', 0.0)):.2f}%)"
    )
    lines.append(
        f"- Multi-hop cascade schemas: `{int(report.get('multi_hop_count', 0))}` "
        f"({100.0 * float(report.get('multi_hop_pct', 0.0)):.2f}%)"
    )
    lines.append(f"- Avg reach depth: `{float(report.get('avg_reach_depth', 0.0)):.3f}`")
    lines.append(
        f"- Avg reachable schemas per schema: `{float(report.get('avg_reachable_schema_count', 0.0)):.3f}`"
    )
    lines.append("")

    lines.append("## Problem-Level Exposure")
    lines.append(f"- Problems analyzed: `{int(report.get('problems_analyzed', 0))}`")
    lines.append(
        f"- Mean redirect exposure ratio: `{float(report.get('problem_redirect_exposure_mean', 0.0)):.3f}`"
    )
    lines.append(
        f"- Mean multi-hop exposure ratio: `{float(report.get('problem_multi_hop_exposure_mean', 0.0)):.3f}`"
    )
    lines.append(
        f"- Problems with any redirect exposure: `{int(report.get('problems_with_redirect_exposure_count', 0))}` "
        f"({100.0 * float(report.get('problems_with_redirect_exposure_pct', 0.0)):.2f}%)"
    )
    lines.append(
        f"- Problems with any multi-hop exposure: `{int(report.get('problems_with_multi_hop_exposure_count', 0))}` "
        f"({100.0 * float(report.get('problems_with_multi_hop_exposure_pct', 0.0)):.2f}%)"
    )
    lines.append("")

    by_family = report.get("by_family", {})
    if isinstance(by_family, dict):
        lines.extend(_render_breakdown_table("By Family", by_family))

    by_depth = report.get("by_problem_depth", {})
    if isinstance(by_depth, dict):
        lines.extend(_render_breakdown_table("By Problem Depth", by_depth))

    lines.append("## Simulation (Corruption Counterfactual)")
    lines.append(f"- Simulated schemas: `{int(report.get('simulation_count', 0))}`")
    lines.append(f"- Successful simulations: `{int(report.get('simulation_ok_count', 0))}`")
    lines.append(
        f"- Corrupted consequent consumed downstream: `{int(report.get('simulation_consumed_count', 0))}` "
        f"({100.0 * float(report.get('simulation_consumed_pct_of_ok', 0.0)):.2f}% of successful)"
    )
    lines.append(
        f"- Answer changed: `{int(report.get('simulation_answer_changed_count', 0))}` "
        f"({100.0 * float(report.get('simulation_answer_changed_pct_of_ok', 0.0)):.2f}% of successful)"
    )
    lines.append(
        f"- Harmful cascades: `{int(report.get('simulation_harmful_count', 0))}` "
        f"({100.0 * float(report.get('simulation_harmful_pct_of_ok', 0.0)):.2f}% of successful)"
    )
    lines.append("")

    status_counts = report.get("simulation_status_counts", {})
    if isinstance(status_counts, dict) and status_counts:
        lines.append("### Simulation Status Counts")
        for k in sorted(status_counts.keys()):
            lines.append(f"- `{k}`: {int(status_counts[k])}")
        lines.append("")

    class_counts = report.get("simulation_class_counts", {})
    if isinstance(class_counts, dict) and class_counts:
        lines.append("### Simulation Classification Counts")
        for k in sorted(class_counts.keys()):
            lines.append(f"- `{k}`: {int(class_counts[k])}")
        lines.append("")

    return "\n".join(lines)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
