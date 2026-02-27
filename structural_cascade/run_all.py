#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Allow execution via `python structural_cascade/run_all.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiments as rex

from structural_cascade.classify_errors import classify_all_schemas
from structural_cascade.report import build_markdown_report, generate_report, write_json
from structural_cascade.structural_analysis import run_structural_analysis
from structural_cascade.simulate_error_propagation import simulate_subset

PAPER_CONFIG_PATH = Path("configs/config_both_modes.json")
PAPER_OUTDIR_BY_DATASET = {
    "ruletaker": "structural_cascade/results_ruletaker",
    "proofwriter": "structural_cascade/results_proofwriter",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structural cascade potential analysis")
    p.add_argument("--config", default="configs/config_both_modes.json")
    p.add_argument("--dataset-key", default="ruletaker", choices=["ruletaker", "proofwriter"])
    p.add_argument("--outdir", default="structural_cascade/results")
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--max-schemas", type=int, default=None)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--simulation-sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    return p.parse_args()


def load_cfg(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_shuffle(problems: List[rex.Problem], seed: int, enabled: bool) -> List[rex.Problem]:
    out = list(problems)
    if enabled:
        rng = random.Random(seed)
        rng.shuffle(out)
    return out


def _assert_paper_locked_args(args: argparse.Namespace) -> None:
    cfg = Path(args.config).as_posix()
    if cfg != PAPER_CONFIG_PATH.as_posix():
        raise ValueError(
            "Paper-locked mode requires --config configs/config_both_modes.json "
            "(recommended: run ./scripts/reproduce.sh)."
        )
    if args.dataset_key not in PAPER_OUTDIR_BY_DATASET:
        raise ValueError("Paper-locked mode requires dataset-key in {ruletaker, proofwriter}.")
    expected_outdir = PAPER_OUTDIR_BY_DATASET[args.dataset_key]
    if Path(args.outdir).as_posix() != expected_outdir:
        raise ValueError(
            f"Paper-locked mode for {args.dataset_key} requires --outdir {expected_outdir}."
        )
    if args.max_problems is not None or args.max_schemas is not None:
        raise ValueError("Paper-locked mode forbids --max-problems and --max-schemas.")
    if not args.simulate:
        raise ValueError("Paper-locked mode requires --simulate.")
    if int(args.simulation_sample) != 60:
        raise ValueError("Paper-locked mode requires --simulation-sample 60.")
    if not args.shuffle:
        raise ValueError("Paper-locked mode requires --shuffle.")
    if int(args.seed) != 42:
        raise ValueError("Paper-locked mode requires --seed 42.")


def main() -> int:
    args = parse_args()
    _assert_paper_locked_args(args)
    t0 = time.time()

    rex.configure_hf_cache_root()

    cfg = load_cfg(Path(args.config))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_problems = rex.load_problems_base(cfg, dataset_key=args.dataset_key)
    base_problems = maybe_shuffle(base_problems, seed=args.seed, enabled=bool(args.shuffle))
    selected_problems = list(
        base_problems[: args.max_problems if args.max_problems is not None else len(base_problems)]
    )

    structural_rows, problem_summaries = run_structural_analysis(
        problems=base_problems,
        max_problems=args.max_problems,
        max_schemas=args.max_schemas,
    )
    write_json(outdir / "structural_analysis.json", structural_rows)
    write_json(outdir / "problem_summaries.json", problem_summaries)

    classified = classify_all_schemas(structural_rows)
    write_json(outdir / "classified_schemas.json", classified)

    simulation_rows: List[Dict[str, object]] = []
    if args.simulate and classified:
        problem_map = {p.problem_id: p for p in selected_problems}
        simulation_rows = simulate_subset(
            classified_rows=classified,
            problem_map=problem_map,
            cfg=cfg,
            seed=args.seed,
            max_items=max(0, int(args.simulation_sample)),
        )
    write_json(outdir / "simulation_results.json", simulation_rows)

    report = generate_report(
        classified_rows=classified,
        problem_summaries=problem_summaries,
        simulation_rows=simulation_rows,
    )
    report["dataset_key"] = args.dataset_key
    report["runtime_seconds"] = round(time.time() - t0, 2)
    report["max_problems"] = args.max_problems
    report["max_schemas"] = args.max_schemas
    report["simulation_count"] = len(simulation_rows)
    write_json(outdir / "report.json", report)

    md = build_markdown_report(report, dataset_key=args.dataset_key)
    (outdir / "report.md").write_text(md + "\n", encoding="utf-8")

    print(f"Saved structural cascade outputs to: {outdir}")
    print(f"Problems analyzed: {len(problem_summaries)}")
    print(f"Schemas analyzed: {len(structural_rows)}")
    print(
        "Cascade potential: "
        f"terminal={report.get('terminal_count', 0)}, "
        f"cascade_eligible={report.get('cascade_eligible_count', 0)}, "
        f"multi_hop={report.get('multi_hop_count', 0)}"
    )
    if args.simulate:
        print(
            "Simulation: "
            f"ok={report.get('simulation_ok_count', 0)}/{len(simulation_rows)}, "
            f"consumed={report.get('simulation_consumed_count', 0)}, "
            f"answer_changed={report.get('simulation_answer_changed_count', 0)}"
        )
    print(f"Runtime: {report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
