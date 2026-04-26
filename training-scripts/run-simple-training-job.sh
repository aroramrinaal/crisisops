#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Run: export HF_TOKEN='<your Hugging Face token>'" >&2
  exit 1
fi

unset HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING="${HF_HUB_DISABLE_EXPERIMENTAL_WARNING:-1}"

hf jobs uv run \
  --token "$HF_TOKEN" \
  --flavor "${HF_JOB_FLAVOR:-h200}" \
  --timeout "${HF_JOB_TIMEOUT:-4h}" \
  -s HF_TOKEN="$HF_TOKEN" \
  --with "trl==0.19.1" \
  --with unsloth \
  --with transformers \
  --with accelerate \
  --with bitsandbytes \
  --with peft \
  --with torch \
  --with datasets \
  --with trackio \
  -e ENV_URL="${ENV_URL:-https://mrinaalarora-crisisops.hf.space}" \
  -e TASK_ID="${TASK_ID:-single_zone_response}" \
  -e MODEL_ID="${MODEL_ID:-unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit}" \
  -e GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-100}" \
  -e HF_REPO_ID="${HF_REPO_ID:-mrinaalarora/crisisops-grpo-easy-lora}" \
  -e TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-mrinaalarora/crisisops-grpo-trackio}" \
  training-scripts/simple-training-script.py
