#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

# Paper-locked reproduction settings (do not override for canonical runs).
CONFIG="configs/config_both_modes.json"
SWEEP_RATES="0.05,0.10,0.20,0.40,0.60"
TAU_VALUES="0.3,0.5,0.7"

export HF_HOME="${HF_HOME:-$(pwd)/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

MAIN_OUT="outputs_main"
SWEEP_OUT="outputs_sweep"
TAU_OUT="outputs_tau_sweep"
SC_RT_OUT="structural_cascade/results_ruletaker"
SC_PW_OUT="structural_cascade/results_proofwriter"

RESULTS_ROOT="results"

mkdir -p "${RESULTS_ROOT}/main" "${RESULTS_ROOT}/sweep" \
  "${RESULTS_ROOT}/tau" "${RESULTS_ROOT}/structural_cascade/ruletaker" "${RESULTS_ROOT}/structural_cascade/proofwriter"

"${PYTHON_BIN}" run_experiments.py --config "${CONFIG}" --outdir "${MAIN_OUT}"
cp "${MAIN_OUT}/results.json" "${RESULTS_ROOT}/main/results.json"
cp "${MAIN_OUT}/report.md" "${RESULTS_ROOT}/main/report.md"
cp "${MAIN_OUT}/dataset_summary.json" "${RESULTS_ROOT}/main/dataset_summary.json"
cp "${MAIN_OUT}/hardware.json" "${RESULTS_ROOT}/main/hardware.json"

"${PYTHON_BIN}" run_experiments.py \
  --config "${CONFIG}" \
  --outdir "${SWEEP_OUT}" \
  --sweep-rates "${SWEEP_RATES}" \
  --skip-ablation --skip-synthetic
cp "${SWEEP_OUT}/sweep_results.json" "${RESULTS_ROOT}/sweep/sweep_results.json"
cp "${SWEEP_OUT}/sweep_report.md" "${RESULTS_ROOT}/sweep/sweep_report.md"

"${PYTHON_BIN}" run_experiments.py \
  --config "${CONFIG}" \
  --outdir "${TAU_OUT}" \
  --tau-sweep "${TAU_VALUES}"
cp "${TAU_OUT}/tau_sweep_results.json" "${RESULTS_ROOT}/tau/tau_sweep_results.json"
cp "${TAU_OUT}/tau_sweep_report.md" "${RESULTS_ROOT}/tau/tau_sweep_report.md"

"${PYTHON_BIN}" structural_cascade/run_all.py \
  --config "${CONFIG}" \
  --dataset-key ruletaker \
  --outdir "${SC_RT_OUT}" \
  --simulate --simulation-sample 60 --shuffle --seed 42
cp "${SC_RT_OUT}/report.json" "${RESULTS_ROOT}/structural_cascade/ruletaker/report.json"
cp "${SC_RT_OUT}/report.md" "${RESULTS_ROOT}/structural_cascade/ruletaker/report.md"

"${PYTHON_BIN}" structural_cascade/run_all.py \
  --config "${CONFIG}" \
  --dataset-key proofwriter \
  --outdir "${SC_PW_OUT}" \
  --simulate --simulation-sample 60 --shuffle --seed 42
cp "${SC_PW_OUT}/report.json" "${RESULTS_ROOT}/structural_cascade/proofwriter/report.json"
cp "${SC_PW_OUT}/report.md" "${RESULTS_ROOT}/structural_cascade/proofwriter/report.md"

echo "Reproduction complete. Artifacts available under ${RESULTS_ROOT}/"
