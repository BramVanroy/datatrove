"""
Progress monitoring example for synthetic data generation.

This example demonstrates how to use the InferenceProgressMonitor to track
generation progress and automatically update a HuggingFace dataset card with
a progress bar and ETA. After inference completes, InferenceDatasetCardGenerator
creates a final dataset card with statistics.

Usage:
    # Local execution (requires GPU)
    python examples/inference/progress_monitoring.py --output-dataset-name my-dataset --local-execution

    # Slurm execution with progress monitoring
    python examples/inference/progress_monitoring.py --output-dataset-name my-dataset --enable-monitoring

    # Slurm execution without progress monitoring
    python examples/inference/progress_monitoring.py --output-dataset-name my-dataset

    # Override cluster-specific Slurm settings explicitly or via DATATROVE_SLURM_* env vars
    python examples/inference/progress_monitoring.py \
        --output-dataset-name my-dataset \
        --gpu-partition gpu_a100 \
        --cpu-partition cpu \
        --account my_project
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.inference.dataset_card_generator import (
    InferenceDatasetCardGenerator,
    InferenceDatasetCardParams,
)
from datatrove.pipeline.inference.progress_monitor import InferenceProgressMonitor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceResult, InferenceRunner
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.logging import logger


EXAMPLES_INFERENCE_DIR = str(Path(__file__).parent)
sys.path.insert(0, EXAMPLES_INFERENCE_DIR)
from utils import (  # noqa: E402
    DEFAULT_SLURM_ACCOUNT,
    DEFAULT_SLURM_CPU_PARTITION,
    DEFAULT_SLURM_GPU_PARTITION,
    DEFAULT_SLURM_TMPDIR,
    DEFAULT_SLURM_VENV_PATH,
    ENV_SLURM_ACCOUNT,
    ENV_SLURM_CPU_PARTITION,
    ENV_SLURM_GPU_PARTITION,
    ENV_SLURM_TMPDIR,
    ENV_SLURM_VENV_PATH,
    check_hf_auth,
    ensure_repo_exists,
    resolve_repo_id,
    resolve_string_setting,
)


# =============================================================================
# Hardcoded configuration - modify these for your use case
# =============================================================================
INPUT_DATASET = "simplescaling/s1K-1.1"
INPUT_SPLIT = "train"
PROMPT_COLUMN = "question"
MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 2048
EXAMPLES_PER_CHUNK = 500
OUTPUT_DIR = "data"


# =============================================================================
# Rollout function
# =============================================================================
async def simple_rollout(
    document: Document,
    generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
) -> InferenceResult:
    """Basic rollout that sends a single request per document."""
    # Note: Using hardcoded value instead of global MAX_TOKENS because globals
    # aren't captured when the function is pickled for Slurm execution
    return await generate(
        {
            "messages": [{"role": "user", "content": document.text}],
            "max_tokens": 2048,
        }
    )


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data with progress monitoring")
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        required=True,
        help="Output HuggingFace dataset name (e.g., 'my-dataset' or 'username/my-dataset')",
    )
    parser.add_argument(
        "--local-execution",
        action="store_true",
        help="Run locally instead of on Slurm (requires GPU)",
    )
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        help="Enable progress monitoring (Slurm only, updates dataset card periodically)",
    )
    parser.add_argument("--account", type=str, default=None, help=f"Slurm account (env: {ENV_SLURM_ACCOUNT})")
    parser.add_argument(
        "--gpu-partition",
        type=str,
        default=None,
        help=f"GPU Slurm partition for inference jobs (env: {ENV_SLURM_GPU_PARTITION})",
    )
    parser.add_argument(
        "--cpu-partition",
        type=str,
        default=None,
        help=f"CPU Slurm partition for monitor/datacard jobs (env: {ENV_SLURM_CPU_PARTITION})",
    )
    parser.add_argument(
        "--venv-path",
        type=str,
        default=None,
        help=f"Virtualenv activate path for Slurm jobs (env: {ENV_SLURM_VENV_PATH})",
    )
    parser.add_argument(
        "--tmpdir",
        type=str,
        default=None,
        help=f"Shared TMPDIR for Slurm jobs (env: {ENV_SLURM_TMPDIR})",
    )
    args = parser.parse_args()

    # Check authentication and resolve repo
    check_hf_auth()
    full_repo_id = resolve_repo_id(args.output_dataset_name)
    ensure_repo_exists(full_repo_id)
    logger.info(f"Output dataset: https://huggingface.co/datasets/{full_repo_id}")

    # Setup paths
    model_safe = MODEL.replace("/", "_")
    final_output_dir = os.path.join(OUTPUT_DIR, model_safe)
    logs_dir = os.path.join(final_output_dir, "logs")
    inference_logs_path = os.path.join(logs_dir, "inference")
    monitor_logs_path = os.path.join(logs_dir, "monitor")
    datacard_logs_path = os.path.join(logs_dir, "datacard")
    checkpoints_path = os.path.join(final_output_dir, "checkpoints")
    stats_path = os.path.join(inference_logs_path, "stats.json")

    # Dataset card parameters (shared between monitor and generator)
    dataset_card_params = InferenceDatasetCardParams(
        output_repo_id=full_repo_id,
        input_dataset_name=INPUT_DATASET,
        input_dataset_split=INPUT_SPLIT,
        input_dataset_config=None,
        prompt_column=PROMPT_COLUMN,
        prompt_template=None,
        prompt_template_name="default",
        system_prompt=None,
        model_name=MODEL,
        model_revision="main",
        generation_kwargs={"max_tokens": MAX_TOKENS},
        spec_config=None,
        stats_path=stats_path,
    )

    # Inference pipeline
    inference_pipeline = [
        HuggingFaceDatasetReader(
            dataset=INPUT_DATASET,
            dataset_options={"split": INPUT_SPLIT},
            text_key=PROMPT_COLUMN,
        ),
        InferenceRunner(
            rollout_fn=simple_rollout,
            config=InferenceConfig(
                server_type="vllm",
                model_name_or_path=MODEL,
                model_kwargs={
                    "max_num_seqs": 500,
                    "enforce-eager": True,
                },  # enforce-eager avoids compile cache conflicts
                server_log_folder=os.path.join(inference_logs_path, "server_logs"),
            ),
            records_per_chunk=EXAMPLES_PER_CHUNK,
            checkpoints_local_dir=checkpoints_path,
            output_writer=ParquetWriter(  # Streams to HF for real-time progress
                output_folder=f"hf://datasets/{full_repo_id}",
                output_filename="data/${rank}_${chunk_index}.parquet",
                expand_metadata=True,
                max_file_size=1024 * 1024,  # ~1MB so we can see progress in real time
                batch_size=10,
            ),
        ),
    ]

    # Monitor pipeline (updates progress to dataset card)
    monitor_pipeline = [
        InferenceProgressMonitor(
            params=dataset_card_params,
            update_interval=60,  # Every minute so we can see progress in real time
        )
    ]

    # Dataset card pipeline (generates final card after inference)
    datacard_pipeline = [InferenceDatasetCardGenerator(params=dataset_card_params)]

    if args.local_execution:
        # Local execution
        inference_executor = LocalPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=inference_logs_path,
            tasks=1,
        )
        datacard_executor = LocalPipelineExecutor(
            pipeline=datacard_pipeline,
            logging_dir=datacard_logs_path,
            tasks=1,
        )

        logger.info("Running inference locally...")
        inference_executor.run()
        logger.info("Generating dataset card...")
        datacard_executor.run()
        logger.info(f"Done! Check: https://huggingface.co/datasets/{full_repo_id}")
    else:
        # Slurm execution
        resolved_account = resolve_string_setting(args.account, ENV_SLURM_ACCOUNT, DEFAULT_SLURM_ACCOUNT)
        resolved_gpu_partition = resolve_string_setting(
            args.gpu_partition, ENV_SLURM_GPU_PARTITION, DEFAULT_SLURM_GPU_PARTITION
        )
        resolved_cpu_partition = resolve_string_setting(
            args.cpu_partition, ENV_SLURM_CPU_PARTITION, DEFAULT_SLURM_CPU_PARTITION
        )
        resolved_venv_path = resolve_string_setting(args.venv_path, ENV_SLURM_VENV_PATH, DEFAULT_SLURM_VENV_PATH)
        resolved_tmpdir = resolve_string_setting(args.tmpdir, ENV_SLURM_TMPDIR, DEFAULT_SLURM_TMPDIR)

        os.makedirs(resolved_tmpdir, exist_ok=True)
        os.environ["TMPDIR"] = resolved_tmpdir
        xet_cache = (
            ' && export HF_XET_CACHE="${TMPDIR}/hf_xet/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_PROCID}"'
            ' && mkdir -p "$HF_XET_CACHE"'
        )
        slurm_env_command = (
            f"export TMPDIR={resolved_tmpdir}"
            f" && source {resolved_venv_path}"
            f" && export PYTHONPATH={EXAMPLES_INFERENCE_DIR}:$PYTHONPATH"
            + xet_cache
        )
        sbatch_args = {"account": resolved_account}

        inference_executor = SlurmPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=inference_logs_path,
            tasks=10,
            workers=4,
            cpus_per_task=11,
            gpus_per_task=1,
            nodes_per_task=1,
            time="12:00:00",
            partition=resolved_gpu_partition,
            job_name="inference",
            qos="",
            sbatch_args=sbatch_args,
            env_command=slurm_env_command,
            venv_path=resolved_venv_path,
        )
        inference_executor.run()

        if args.enable_monitoring:
            # Update monitor with inference job ID to stop if inference fails
            monitor_pipeline[0].inference_job_id = inference_executor.job_id

            monitor_executor = SlurmPipelineExecutor(
                pipeline=monitor_pipeline,
                logging_dir=monitor_logs_path,
                tasks=1,
                time="7-00:00:00",
                partition=resolved_cpu_partition,
                job_name="monitor",
                qos="",
                sbatch_args=sbatch_args,
                env_command=slurm_env_command,
                venv_path=resolved_venv_path,
            )
            monitor_executor.run()
            logger.info(f"Monitor job submitted: {monitor_executor.job_id}")

        datacard_executor = SlurmPipelineExecutor(
            pipeline=datacard_pipeline,
            logging_dir=datacard_logs_path,
            tasks=1,
            time="00:10:00",
            partition=resolved_cpu_partition,
            depends=inference_executor,
            job_name="datacard",
            qos="",
            sbatch_args=sbatch_args,
            env_command=slurm_env_command,
            venv_path=resolved_venv_path,
        )
        datacard_executor.run()

        logger.info("Jobs submitted!")
        logger.info(f"  Inference job: {inference_executor.job_id}")
        logger.info(f"  Datacard job: {datacard_executor.job_id}")
        logger.info(f"Check: https://huggingface.co/datasets/{full_repo_id}")


if __name__ == "__main__":
    main()
