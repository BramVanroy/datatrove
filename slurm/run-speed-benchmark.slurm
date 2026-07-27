#!/bin/bash
# Runs a speed benchmark orchestrated by datatrove, based on the yaml config file in
# datatrove/examples/inference/benchmark/throughput-benchmark-dutch.yaml. The benchmark will
# run a series of SLURM jobs to benchmark the throughput of various models on the Dutch language.
# This process is done in two tiers: first one is focused on hyperparameters such as number of sequences,
# number of tokens, number of GPUs, etc. After settling on those, a second step measures the effect
# of multi-token prediction (MTP) to get "the best out of the models".
#SBATCH --job-name=benchmark
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=tnsr72764
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=2

set -e

module load 2025
module load git/2.49.0-GCCcore-14.2.0
module load GCC/14.2.0

DATATROVE_DIR="$HOME/synthetic-data-pipeline/datatrove"

SCRATCH_DIR="/scratch-shared/$USER"
SCRATCH_TMP_DIR="$SCRATCH_DIR/tmp"
SCRATCH_VENV_DIR="$SCRATCH_DIR/venv"

mkdir -p "$SCRATCH_TMP_DIR" "$SCRATCH_VENV_DIR"

# Ensure that the datatrove venv is set up and activated on scratch
export VENV_PATH="$SCRATCH_VENV_DIR/datatrove-venv"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

# Qwen MoE seem to struggle with flashinfer backend, may especially be an issue when TP>1.
# Note that this is not an official datatrove envvar but an addition I made in vllm_server.py.
export VLLM_TRITON_GDN_BACKEND=1

cd "$DATATROVE_DIR"

# --- should only be done once. Uncomment this block if it is the first time you run this benchmark on this machine.
# uv sync --frozen --extra inference,ray,multilingual,processing,io,cli --no-dev -p 3.13

# MODELS=(
#     "google/gemma-4-E4B-it-qat-w4a16-ct"
#     "Qwen/Qwen3.5-4B"
#     "ibm-granite/granite-4.1-3b-fp8"
#     "mistralai/Ministral-3-3B-Instruct-2512"
#     "ibm-granite/granite-4.1-8b-fp8"
#     "Qwen/Qwen3.5-9B"
#     "utter-project/EuroLLM-9B-Instruct-2512"
#     "swiss-ai/Apertus-8B-Instruct-2509"
#     "mistralai/Ministral-3-8B-Instruct-2512"
#     "google/gemma-4-12B-it-qat-w4a16-ct"
#     "google/gemma-4-26B-A4B-it"
#     "Qwen/Qwen3.6-35B-A3B-FP8"
#     # Assistant (Google only. Qwen uses NEXT; others uses ngram/suff)
#     "google/gemma-4-E4B-it-assistant"
#     "google/gemma-4-12B-it-assistant"
#     "google/gemma-4-26B-A4B-it-assistant"
# )

# echo "=== Starting model downloads ==="
# for MODEL in "${MODELS[@]}"; do
#     echo "Downloading model: $MODEL"
#     uv run hf download "$MODEL"
# done

# echo "=== Downloading dataset ==="
# uv run python -c 'from datasets import load_dataset; load_dataset("epfml/FineWeb2-HQ", "nld_Latn", split="train")'
# ---

# ! make sure that models and datasets have been pre-downloaded
# We do it here to prevent race conditions since different proccesses
# will try to download the same model, so best to pre-download
export HF_HUB_OFFLINE=1

ARGS=(
    --config "$DATATROVE_DIR/examples/inference/benchmark/throughput-benchmark-dutch.yaml"
    --skip-failure-reasons "OOM"
#    --dry-run
)

# 'launch experiments' will launch the required GPU SLURM jobs
uv run python "$DATATROVE_DIR/examples/inference/benchmark/launch_experiments.py" "${ARGS[@]}"
