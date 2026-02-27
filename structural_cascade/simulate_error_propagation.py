from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import run_experiments as rex


def _oracle_estimator() -> rex.EstimatorModel:
    return rex.EstimatorModel(
        threshold=0.5,
        backend="oracle_all_reliable",
        source_scores={},
        family_scores={},
        global_score=1.0,
        metrics={},
    )


def choose_redirect_target(problem: rex.Problem, schema_index: int, rng: random.Random) -> Optional[str]:
    """Match run_experiments redirect corruption candidate logic."""
    fact_props = {rex.normalize_prop(f) for f in problem.facts if rex.normalize_prop(f)}
    antecedent_counts: Dict[str, int] = {}
    for cand_schema in problem.schemas:
        for ant in cand_schema.antecedents:
            ant_n = rex.normalize_prop(ant)
            if ant_n:
                antecedent_counts[ant_n] = antecedent_counts.get(ant_n, 0) + 1

    schema = problem.schemas[schema_index]
    original_consequent = rex.normalize_prop(schema.consequent)
    redirect_candidates: List[Tuple[int, str]] = []

    for j, cand_schema in enumerate(problem.schemas):
        if j == schema_index:
            continue
        for ant in cand_schema.antecedents:
            ant_n = rex.normalize_prop(ant)
            if not ant_n:
                continue
            if ant_n == original_consequent:
                continue
            if ant_n == rex.negate_prop(original_consequent):
                continue
            if ant_n.startswith("not|"):
                continue
            score = antecedent_counts.get(ant_n, 1)
            if ant_n not in fact_props:
                score += 100
            redirect_candidates.append((score, ant_n))

    if not redirect_candidates:
        return None

    redirect_candidates.sort(key=lambda x: (-x[0], x[1]))
    top_k = min(5, len(redirect_candidates))
    choice_idx = rng.randrange(top_k)
    return redirect_candidates[choice_idx][1]


def _find_schema_index(problem: rex.Problem, schema_id: str) -> Optional[int]:
    for i, s in enumerate(problem.schemas):
        if s.schema_id == schema_id:
            return i
    return None


def _schema_node_ids(result: rex.InferenceResult, schema_id: str) -> set[str]:
    out: set[str] = set()
    for nid, node in result.nodes.items():
        rid = node.rule_id
        if rid is None:
            continue
        if rid == schema_id or rid.startswith(f"{schema_id}@"):
            out.add(nid)
    return out


def simulate_schema_corruption(
    schema_row: Dict[str, object],
    problem: rex.Problem,
    cfg: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    """Counterfactual: baseline problem vs same problem with schema consequent redirected."""
    sid = str(schema_row.get("schema_id", ""))

    baseline = copy.deepcopy(problem)
    corrupted = copy.deepcopy(problem)

    idx = _find_schema_index(corrupted, sid)
    if idx is None:
        return {
            "problem_id": problem.problem_id,
            "schema_id": sid,
            "simulation_status": "schema_not_found",
        }

    rng = random.Random(rex.deterministic_seed("structural_redirect", seed, problem.problem_id, sid))
    replacement = choose_redirect_target(corrupted, idx, rng)
    if replacement is None:
        return {
            "problem_id": problem.problem_id,
            "schema_id": sid,
            "simulation_status": "no_redirect_target",
        }

    rex.corrupt_schema_redirect(corrupted.schemas[idx], replacement)

    estimator = _oracle_estimator()
    base_res = rex.infer_problem(
        problem=baseline,
        estimator=estimator,
        model="instance_weighted",
        cfg=cfg,
        seed=seed,
    )
    corr_res = rex.infer_problem(
        problem=corrupted,
        estimator=estimator,
        model="instance_weighted",
        cfg=cfg,
        seed=seed,
    )

    base_props = set(base_res.best_committed.keys())
    corr_props = set(corr_res.best_committed.keys())
    additional_props = sorted(corr_props - base_props)

    corr_schema_node_ids = _schema_node_ids(corr_res, sid)
    consumed_nodes = 0
    for node in corr_res.nodes.values():
        if node.rule_id is None:
            continue
        if any(pid in corr_schema_node_ids for pid in node.parent_ids):
            consumed_nodes += 1

    base_pred = int(base_res.predicted_label)
    corr_pred = int(corr_res.predicted_label)
    gold = int(problem.answer)

    base_correct = base_pred == gold
    corr_correct = corr_pred == gold

    if consumed_nodes == 0:
        sim_class = "terminal-like"
    elif base_correct and not corr_correct:
        sim_class = "harmful-cascade"
    else:
        sim_class = "propagating-but-nonharmful"

    return {
        "problem_id": problem.problem_id,
        "schema_id": sid,
        "simulation_status": "ok",
        "replacement_consequent": replacement,
        "additional_derivations": len(additional_props),
        "additional_derivation_examples": additional_props[:8],
        "corrupted_consequent_consumed": consumed_nodes > 0,
        "consuming_node_count": consumed_nodes,
        "simulation_classification": sim_class,
        "answer_baseline": base_pred,
        "answer_corrupted": corr_pred,
        "gold_answer": gold,
        "answer_changed": base_pred != corr_pred,
        "baseline_correct": base_correct,
        "corrupted_correct": corr_correct,
    }


def simulate_subset(
    classified_rows: List[Dict[str, object]],
    problem_map: Dict[str, rex.Problem],
    cfg: Dict[str, object],
    seed: int,
    max_items: int,
) -> List[Dict[str, object]]:
    candidates = [r for r in classified_rows if int(r.get("reach_depth", 0)) >= 1]
    if not candidates:
        candidates = list(classified_rows)

    out: List[Dict[str, object]] = []
    for row in candidates[: max(0, int(max_items))]:
        pid = str(row.get("problem_id", ""))
        p = problem_map.get(pid)
        if p is None:
            continue
        sim = simulate_schema_corruption(row, p, cfg=cfg, seed=seed)
        merged = dict(row)
        merged.update({f"simulation_{k}": v for k, v in sim.items()})
        out.append(merged)
    return out
