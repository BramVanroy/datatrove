#!/bin/bash
#SBATCH --job-name=benchmark-analysis
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=tnsr72764
#SBATCH --partition=genoa,rome
#SBATCH --cpus-per-task=24

source "$HOME/.bashrc"

set -e

load_modules
set_cache_dirs

export DATATROVE_DIR="/projects/prjs0986/embedding/synth-data/datatrove"

export SCRATCH_DIR="/scratch-shared/bvanroy"
export VENV_PATH="$SCRATCH_DIR/venv/venv-datatrove"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

cd "$DATATROVE_DIR"

if [[ ! -d "$VENV_PATH" ]]; then
    # Installs in editable mode
    uv sync --frozen --extra inference,ray,multilingual,processing,io,cli --no-dev -p 3.13
fi

source "$VENV_PATH/bin/activate"

export HF_HUB_OFFLINE=1

ARGS=(
    --root "/projects/prjs0986/embedding/synth-data/benchmark-results"
)

export TMPDIR="/scratch-shared/bvanroy/tmp/datatrove-tmp"

python "$DATATROVE_DIR/examples/inference/benchmark/analyze_results.py" "${ARGS[@]}"