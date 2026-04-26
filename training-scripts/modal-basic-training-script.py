"""
Thin Modal wrapper that runs training-scripts/simple-training-script.py
on Modal's cloud GPUs, exactly like run-simple-training-job.sh does on HF Jobs.

Usage:
    modal run training-scripts/modal-basic-training-script.py

    # With overrides:
    modal run training-scripts/modal-basic-training-script.py \
        --grpo-max-steps 200 \
        --task-id multi_zone_triage

Prerequisites:
    - modal token set (authenticated CLI)
    - A Modal Secret named "huggingface-secret" (type: Hugging Face)
"""

import modal
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Image — same packages as the HF Job, plus the training script itself
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "trl==0.19.1",
        "unsloth",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "torch",
        "datasets",
        "trackio",
    )
    # Copy the existing training script into the container.
    # The source path is relative to the directory where `modal run` is invoked.
    .add_local_file("training-scripts/simple-training-script.py", "/root/simple-training-script.py")
)

app = modal.App("crisisops-grpo-basic", image=image)


# ---------------------------------------------------------------------------
# Remote function — mirrors hf jobs uv run ... training-scripts/simple-training-script.py
# ---------------------------------------------------------------------------

@app.function(
    gpu="H200",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=8,
    memory=32768,
    ephemeral_disk=262144,
    timeout=4 * 60 * 60,
)
def run_training(
    env_url: str = "https://mrinaalarora-crisisops.hf.space",
    task_id: str = "single_zone_response",
    model_id: str = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
    grpo_max_steps: int = 100,
    grpo_per_device_batch: int = 3,
    hf_repo_id: str = "mrinaalarora/crisisops-grpo-easy-lora",
    trackio_space_id: str = "mrinaalarora/crisisops-grpo-trackio",
) -> None:
    """
    Set the same env vars that run-simple-training-job.sh sets, then exec
    simple-training-script.py unchanged.
    """
    os.environ["ENV_URL"] = env_url
    os.environ["TASK_ID"] = task_id
    os.environ["MODEL_ID"] = model_id
    os.environ["GRPO_MAX_STEPS"] = str(grpo_max_steps)
    os.environ["GRPO_PER_DEVICE_BATCH"] = str(grpo_per_device_batch)
    os.environ["HF_REPO_ID"] = hf_repo_id
    os.environ["TRACKIO_SPACE_ID"] = trackio_space_id
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    # HF_TOKEN is injected automatically by modal.Secret.from_name("huggingface-secret")

    subprocess.run([sys.executable, "/root/simple-training-script.py"], check=True)


# ---------------------------------------------------------------------------
# Local entrypoint — modal run ... maps CLI flags to the remote function
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    env_url: str = "https://mrinaalarora-crisisops.hf.space",
    task_id: str = "single_zone_response",
    model_id: str = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
    grpo_max_steps: int = 100,
    grpo_per_device_batch: int = 3,
    hf_repo_id: str = "mrinaalarora/crisisops-grpo-easy-lora",
    trackio_space_id: str = "mrinaalarora/crisisops-grpo-trackio",
):
    run_training.remote(
        env_url=env_url,
        task_id=task_id,
        model_id=model_id,
        grpo_max_steps=grpo_max_steps,
        grpo_per_device_batch=grpo_per_device_batch,
        hf_repo_id=hf_repo_id,
        trackio_space_id=trackio_space_id,
    )
