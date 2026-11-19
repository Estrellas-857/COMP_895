#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
r="$PWD/experiments/cse805_random_forest_v1/runtime"
export PYTHONPATH="$PWD/experiments/cse805_error_prediction_v1/runtime/packages:$PWD"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$r/matplotlib" CUDA_CACHE_PATH="$r/cuda" TRITON_CACHE_DIR="$r/triton"
mkdir -p "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
phase="${1:?expected preflight, train, evaluate, report or verify}"
shift
case "$phase" in preflight|train|evaluate|report|verify) ;; *) exit 2 ;; esac
exec .venv/bin/python -B -u -m "experiments.cse805_random_forest_v1.$phase" "$@"
