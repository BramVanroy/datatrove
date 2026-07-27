#!/bin/bash
# Runs an analysis over benchmark results retrieved by run-speed-benchmark.slurm
# which live in datatrove/benchmark-results. The analysis will generate a summary of the results and
# save those to datatrove/benchmark-results/*.csv files
# The SLURM output .log file will also contain a summary table.
#SBATCH --job-name=benchmark-analysis
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=tnsr72764
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24

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
# Datatrove uses tempfiles to move SLURM files around but they
# may get deleted between jobs for us, so we stabilize the temp dir here
# to a non-auto delete tmp
export TMPDIR="$SCRATCH_TMP_DIR/datatrove-tmp"

cd "$DATATROVE_DIR"

uv sync --frozen --extra inference,ray,multilingual,processing,io,cli --no-dev -p 3.13

export HF_HUB_OFFLINE=1

ARGS=(
    --root "$DATATROVE_DIR/benchmark-results"
)

uv run python "$DATATROVE_DIR/examples/inference/benchmark/analyze_results.py" "${ARGS[@]}"
