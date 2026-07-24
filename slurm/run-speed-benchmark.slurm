#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=tnsr72764
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=2

# also loads env vars like SCRATCH_DIR
source "$HOME/.bashrc"

set -e

load_modules

# Will ONLY do data/model preloading. After that is finished,
# set to 0 and re-run
DO_PRELOAD=1

DATATROVE_DIR="$GPTNLE_ROOT/subtasks/synth-data/datatrove"

# Ensure that the datatrove venv is set up and activated on scratch
export VENV_PATH="$SCRATCH_DIR/venv/venv-datatrove"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"
# Datatrove uses tempfiles to move SLURM files around but they
# may get deleted between jobs for us, so we stabilize the temp dir here
# to a non-auto delete tmp
export TMPDIR="$SCRATCH_TMP_DIR/tmp-datatrove"

cd "$DATATROVE_DIR"

if [[ ! -d "$VENV_PATH" ]]; then
    # Installs in editable mode in UV_PROJECT_ENVIRONMENT
    uv sync --frozen --extra inference,ray,multilingual,processing,io,cli --no-dev -p 3.13
fi

source "$VENV_PATH/bin/activate"

# ! make sure that models and datasets have been pre-downloaded
# We do it here to prevent race conditions since different proccesses
# will try to download the same model, so best to pre-download
if (( DO_PRELOAD == 1 )); then
     MODELS=(
        "google/gemma-4-E4B-it-qat-w4a16-ct"
        "Qwen/Qwen3.5-4B"
        "ibm-granite/granite-4.1-3b-fp8"
        "mistralai/Ministral-3-3B-Instruct-2512"
        "ibm-granite/granite-4.1-8b-fp8"
        "Qwen/Qwen3.5-9B"
        "utter-project/EuroLLM-9B-Instruct-2512"
        "swiss-ai/Apertus-8B-Instruct-2509"
        "mistralai/Ministral-3-8B-Instruct-2512"
        "google/gemma-4-12B-it-qat-w4a16-ct"
        "google/gemma-4-26B-A4B-it"
        "Qwen/Qwen3.6-35B-A3B-FP8"
        # Assistant (Qwen uses NEXT; others uses ngram/suff)
        "google/gemma-4-E4B-it-assistant"
        "google/gemma-4-12B-it-assistant"
        "google/gemma-4-26B-A4B-it-assistant"
    )

    echo "=== Starting model downloads ==="
    for MODEL in "${MODELS[@]}"; do
        echo "Downloading model: $MODEL"
        hf download "$MODEL"
    done

    # echo "=== Downloading dataset ==="
    # python -c 'from datasets import load_dataset; load_dataset("epfml/FineWeb2-HQ", "nld_Latn", split="train")'
    # echo "Preloading completed. Change DO_PRELOAD=0 and re-run."
    # exit 0
fi

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
python "$DATATROVE_DIR/examples/inference/benchmark/launch_experiments.py" "${ARGS[@]}"
