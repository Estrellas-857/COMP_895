#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
ext="$PWD/experiments/cse805_class_msp_interaction_v1"
export PYTHONPATH="$PWD/experiments/cse805_error_prediction_v1/runtime/packages:$PWD"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 MPLCONFIGDIR="$ext/runtime/matplotlib" CUDA_CACHE_PATH="$ext/runtime/cuda-cache"
mkdir -p "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
phase="${1:?expected preflight, train, evaluate, report or verify}"
shift
case "$phase" in preflight|train|evaluate|report|verify) ;; *) exit 2 ;; esac
exec .venv/bin/python -B -u -m "experiments.cse805_class_msp_interaction_v1.$phase" "$@"
