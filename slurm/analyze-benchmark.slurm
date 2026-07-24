#!/bin/bash
#SBATCH --job-name=benchmark-analysis
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=tnsr72764
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24

# also loads env vars like SCRATCH_DIR
source "$HOME/.bashrc"

set -e

load_modules

DATATROVE_DIR="$GPTNLE_ROOT/subtasks/synth-data/datatrove"

# Ensure that the datatrove venv is set up and activated on scratch
export VENV_PATH="$SCRATCH_DIR/venv/venv-datatrove"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"
# Datatrove uses tempfiles to move SLURM files around but they
# may get deleted between jobs for us, so we stabilize the temp dir here
# to a non-auto delete tmp
export TMPDIR="$SCRATCH_TMP_DIR/datatrove-tmp"

cd "$DATATROVE_DIR"

if [[ ! -d "$VENV_PATH" ]]; then
    # Installs in editable mode in UV_PROJECT_ENVIRONMENT
    uv sync --frozen --extra inference,ray,multilingual,processing,io,cli --no-dev -p 3.13
fi

source "$VENV_PATH/bin/activate"

export HF_HUB_OFFLINE=1

ARGS=(
    # Set in the benchmark yaml file
    --root "/home/bvanroy/gpt-nl-e/subtasks/synth-data/datatrove/benchmark-results"
)

python "$DATATROVE_DIR/examples/inference/benchmark/analyze_results.py" "${ARGS[@]}"
