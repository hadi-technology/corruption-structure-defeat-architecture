from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import run_experiments as rex


def build_schema_graph(problem: rex.Problem) -> Dict[str, Set[str]]:
    """Directed graph A->B if A's consequent appears in any antecedent of B."""
    consequent_to_schema_ids: Dict[str, List[str]] = {}
    for schema in problem.schemas:
        cons = rex.normalize_prop(schema.consequent)
        if not cons:
            continue
        consequent_to_schema_ids.setdefault(cons, []).append(schema.schema_id)

    adjacency: Dict[str, Set[str]] = {s.schema_id: set() for s in problem.schemas}
    for dst in problem.schemas:
        dst_id = dst.schema_id
        for ant in dst.antecedents:
            ant_n = rex.normalize_prop(ant)
            if not ant_n:
                continue
            for src_id in consequent_to_schema_ids.get(ant_n, []):
                if src_id != dst_id:
                    adjacency.setdefault(src_id, set()).add(dst_id)
    return adjacency


def forward_reach_depth(adjacency: Dict[str, Set[str]], start: str) -> Tuple[int, int]:
    """Returns (max_hops, reachable_schema_count excluding self)."""
    if start not in adjacency:
        return 0, 0

    q: List[Tuple[str, int]] = [(start, 0)]
    seen: Dict[str, int] = {start: 0}

    for node, depth in q:
        for nxt in adjacency.get(node, set()):
            if nxt in seen:
                continue
            seen[nxt] = depth + 1
            q.append((nxt, depth + 1))

    max_hops = max(seen.values()) if seen else 0
    reachable_count = max(0, len(seen) - 1)
    return max_hops, reachable_count


def analyze_problem(problem: rex.Problem) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    adjacency = build_schema_graph(problem)

    rows: List[Dict[str, object]] = []
    terminal = 0
    cascade_eligible = 0
    multi_hop = 0

    for schema in problem.schemas:
        max_hops, reachable_count = forward_reach_depth(adjacency, schema.schema_id)
        has_downstream = len(adjacency.get(schema.schema_id, set())) > 0

        row = {
            "problem_id": problem.problem_id,
            "problem_depth": int(problem.depth),
            "schema_id": schema.schema_id,
            "schema_family": schema.family,
            "schema_text": schema.text,
            "antecedent_propositions": [rex.normalize_prop(a) for a in schema.antecedents],
            "consequent_proposition": rex.normalize_prop(schema.consequent),
            "downstream_out_degree": len(adjacency.get(schema.schema_id, set())),
            "reach_depth": int(max_hops),
            "reachable_schema_count": int(reachable_count),
            "has_downstream_consumer": bool(has_downstream),
        }
        rows.append(row)

        if max_hops == 0:
            terminal += 1
        else:
            cascade_eligible += 1
        if max_hops >= 2:
            multi_hop += 1

    schema_total = len(problem.schemas)
    problem_summary = {
        "problem_id": problem.problem_id,
        "problem_depth": int(problem.depth),
        "schema_count": schema_total,
        "redirect_exposure": cascade_eligible,
        "redirect_exposure_ratio": cascade_eligible / max(1, schema_total),
        "multi_hop_exposure": multi_hop,
        "multi_hop_exposure_ratio": multi_hop / max(1, schema_total),
        "terminal_count": terminal,
    }
    return rows, problem_summary


def run_structural_analysis(
    problems: Sequence[rex.Problem],
    max_problems: Optional[int] = None,
    max_schemas: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    selected = list(problems[: max_problems if max_problems is not None else len(problems)])

    rows: List[Dict[str, object]] = []
    problem_summaries: List[Dict[str, object]] = []

    for p in selected:
        p_rows, p_summary = analyze_problem(p)
        if max_schemas is not None and len(rows) + len(p_rows) > max_schemas:
            keep = max(0, int(max_schemas) - len(rows))
            p_rows = p_rows[:keep]
            # Recompute truncated summary if max_schemas clips this problem.
            schema_total = len(p_rows)
            cascade_eligible = sum(1 for r in p_rows if int(r.get("reach_depth", 0)) >= 1)
            multi_hop = sum(1 for r in p_rows if int(r.get("reach_depth", 0)) >= 2)
            terminal = sum(1 for r in p_rows if int(r.get("reach_depth", 0)) == 0)
            p_summary = {
                "problem_id": p.problem_id,
                "problem_depth": int(p.depth),
                "schema_count": schema_total,
                "redirect_exposure": cascade_eligible,
                "redirect_exposure_ratio": cascade_eligible / max(1, schema_total),
                "multi_hop_exposure": multi_hop,
                "multi_hop_exposure_ratio": multi_hop / max(1, schema_total),
                "terminal_count": terminal,
                "truncated": True,
            }

        rows.extend(p_rows)
        problem_summaries.append(p_summary)

        if max_schemas is not None and len(rows) >= max_schemas:
            break

    return rows, problem_summaries


def build_problem_schema_views(problems: Sequence[rex.Problem]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for p in problems:
        rows: List[Dict[str, object]] = []
        for s in p.schemas:
            rows.append(
                {
                    "schema_id": s.schema_id,
                    "schema_family": s.family,
                    "antecedent_propositions": [rex.normalize_prop(a) for a in s.antecedents],
                    "consequent_proposition": rex.normalize_prop(s.consequent),
                }
            )
        out[p.problem_id] = rows
    return out
