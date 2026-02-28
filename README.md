# Corruption Structure vs Defeat Architecture

This repository is intentionally minimal: there is one supported command to reproduce the paper results.

## Setup

- Python `>=3.10` (see `pyproject.toml`)
- Install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

If needed in your environment:

```bash
export HF_TOKEN=<your_token>
```

## Reproduce Results

Run exactly:

```bash
./scripts/reproduce.sh
```

This regenerates the canonical paper artifacts under `results/`.

## BoardgameQA (Defeasibility Extension)

BoardgameQA support is implemented as an optional config-driven run (separate from the paper reproduction command):

```bash
python run_experiments.py --config configs/config_boardgameqa.json --outdir outputs_boardgameqa
```

This run:
- uses depth-2/3 instances
- filters to `proved` / `disproved`
- parses tuple-structured facts/rules from `theory`
- applies rule preferences to defeat timing (`IW`, `V1`, `V2`, `Gating`)

## How The Script Works

`./scripts/reproduce.sh` runs the full paper pipeline in this order:

1. Main experiment run (`run_experiments.py`) and writes canonical outputs to `results/main/`.
2. Corruption-rate sweep run and writes canonical outputs to `results/sweep/`.
3. Tau-sweep run and writes canonical outputs to `results/tau/`.
4. Structural-cascade runs for RuleTaker and ProofWriter and writes canonical outputs to:
   `results/structural_cascade/ruletaker/` and `results/structural_cascade/proofwriter/`.

The script uses fixed paper settings (`configs/config_both_modes.json`, fixed sweep rates, fixed tau values, fixed structural-cascade parameters) so runs are consistent with the paper protocol.

## Canonical Artifacts

- `results/main/`
- `results/sweep/`
- `results/tau/`
- `results/structural_cascade/ruletaker/`
- `results/structural_cascade/proofwriter/`
