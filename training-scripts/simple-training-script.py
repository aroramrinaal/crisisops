"""
GRPO training run for CrisisOps + Unsloth on a single H200.

One-step training: each GRPO sample is a fresh CrisisOps reset plus one model
action. The model writes a JSON action, the env applies it, and the env's
per-step reward becomes the GRPO reward. Across a training step we run
`per_device_train_batch_size * gradient_accumulation_steps * num_generations`
of these action requests, GRPO normalizes within each group and updates the
LoRA adapter.

Why step-level instead of episode-level:
- Plays nicely with TRL's default GRPOTrainer contract (one prompt, one
  completion, one reward) instead of the experimental rollout_func path.
- Prompt and reward stay aligned because each completion is scored against the
  same deterministic reset seed that produced its prompt.
- This is deliberately the smallest reliable RL artifact for the hackathon:
  visible loss/reward curves first, fuller episode training later.

The script:
1. Loads Qwen2.5-Coder-3B-Instruct in 4-bit via unsloth + LoRA r=16
2. Builds a HF Dataset of deterministic reset observations for fixed seeds
3. The reward function resets the matching seed, applies the generated action,
   and returns the env reward plus small validity/action-shaping bonuses
4. Logs to trackio so the loss + reward curve are visible from a HF Space.

Usage:

export HF_TOKEN=hf_...
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

hf jobs uv run \
  --token "$HF_TOKEN" \
  --flavor h200 \
  --timeout 4h \
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
  -e ENV_URL=https://mrinaalarora-crisisops.hf.space \
  -e TASK_ID=single_zone_response \
  -e MODEL_ID=unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  -e GRPO_MAX_STEPS=100 \
  -e HF_REPO_ID=mrinaalarora/crisisops-grpo-easy-lora \
  -e TRACKIO_SPACE_ID=mrinaalarora/crisisops-grpo-trackio \
  training-scripts/simple-training-script.py
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import textwrap
import threading
import time
import uuid
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# --------------------------------------------------------------------------- #
# Config (env-var driven so HF Jobs can override without code changes)
# --------------------------------------------------------------------------- #

ENV_URL = os.getenv("ENV_URL", "https://mrinaalarora-crisisops.hf.space").rstrip("/")
MODEL_ID = os.getenv("MODEL_ID", "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit").strip()
TASK_ID = os.getenv("TASK_ID", "single_zone_response").strip()

# Training scale. 100 GRPO steps with the sizing below = 100 weight updates.
# With grad_accum=4, per-device batch=3, and num_generations=4, Unsloth sees
# 3 unique prompts per generation cycle, satisfying its >2 prompt guidance.
GRPO_MAX_STEPS = int(os.getenv("GRPO_MAX_STEPS", "100"))
GRPO_NUM_GENERATIONS = int(os.getenv("GRPO_NUM_GENERATIONS", "4"))
GRPO_PER_DEVICE_BATCH = int(os.getenv("GRPO_PER_DEVICE_BATCH", "3"))
GRPO_GRAD_ACCUM = int(os.getenv("GRPO_GRAD_ACCUM", "4"))
GRPO_LEARNING_RATE = float(os.getenv("GRPO_LEARNING_RATE", "5e-6"))
GRPO_BETA = float(os.getenv("GRPO_BETA", "0.0"))
GRPO_TEMPERATURE = float(os.getenv("GRPO_TEMPERATURE", "0.9"))
GRPO_TOP_P = float(os.getenv("GRPO_TOP_P", "0.95"))
GRPO_MAX_PROMPT_LENGTH = int(os.getenv("GRPO_MAX_PROMPT_LENGTH", "3072"))
GRPO_MAX_COMPLETION_LENGTH = int(os.getenv("GRPO_MAX_COMPLETION_LENGTH", "512"))
GRPO_WARMUP_STEPS = int(os.getenv("GRPO_WARMUP_STEPS", "5"))
GRPO_LOGGING_STEPS = int(os.getenv("GRPO_LOGGING_STEPS", "1"))
GRPO_SAVE_STEPS = int(os.getenv("GRPO_SAVE_STEPS", "25"))
GRPO_LORA_RANK = int(os.getenv("GRPO_LORA_RANK", "16"))
GRPO_MAX_GRAD_NORM = float(os.getenv("GRPO_MAX_GRAD_NORM", "0.1"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "4096"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "crisisops-grpo-easy-lora").strip()
HF_REPO_ID = os.getenv("HF_REPO_ID", "").strip() or None
TRACKIO_SPACE_ID = os.getenv("TRACKIO_SPACE_ID", "").strip() or None
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
RUN_NAME = os.getenv("RUN_NAME", f"crisisops-{TASK_ID}-grpo-one-step").strip()

# Prompt seeds. We repeat a compact set of deterministic reset observations to
# keep the run cheap and low-variance.
SEED_POOL_SIZE = int(os.getenv("SEED_POOL_SIZE", "16"))

# How many step-request rows to put in the dataset. GRPOTrainer iterates over
# these; with num_train_epochs=1 and max_steps capped, we just need enough to
# cover max_steps * per_device_batch * grad_accum. We oversize so the trainer
# never runs out before max_steps.
DATASET_ROWS = int(
    os.getenv(
        "DATASET_ROWS",
        str(max(256, GRPO_MAX_STEPS * GRPO_PER_DEVICE_BATCH * GRPO_GRAD_ACCUM * 2)),
    )
)


warnings.filterwarnings(
    "ignore",
    message=r".*AttentionMaskConverter.*deprecated.*",
    category=FutureWarning,
)

ACTION_JSON_RE = re.compile(r"\{[\s\S]*\}")


# --------------------------------------------------------------------------- #
# Domain constants (mirror baseline_smoke_test_unsloth_crisisops.py so the
# training prompt has the same shape as the baseline that produced 0.760
# on easy / 0.270 on medium).
# --------------------------------------------------------------------------- #

TASK_TIERS: Dict[str, str] = {
    "single_zone_response": "easy",
    "multi_zone_triage": "medium",
    "cascading_crisis": "hard",
    "multi_district_coordination": "expert",
}

TASK_CONFIGS: Dict[str, Dict[str, int]] = {
    "single_zone_response": {"episode_cap": 8, "expected_reports": 3, "stream_done_step": 0},
    "multi_zone_triage": {"episode_cap": 15, "expected_reports": 6, "stream_done_step": 0},
    "cascading_crisis": {"episode_cap": 25, "expected_reports": 10, "stream_done_step": 12},
    "multi_district_coordination": {"episode_cap": 40, "expected_reports": 16, "stream_done_step": 20},
}

INCIDENT_UNIT_TYPES: Dict[str, set[str]] = {
    "flood": {"rescue_team", "evac_bus"},
    "collapse": {"rescue_team", "medical_unit"},
    "medical_surge": {"medical_unit"},
    "fire": {"rescue_team"},
    "contamination": {"supply_truck", "medical_unit"},
    "power_outage": {"supply_truck"},
}

TASK_BY_UNIT_TYPE = {
    "rescue_team": "rescue",
    "medical_unit": "medical",
    "supply_truck": "supply_delivery",
    "evac_bus": "evacuation",
    "recon_drone": "recon",
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an emergency operations commander for CrisisOps. Return exactly one
    JSON object for the next environment action.

    Use only IDs that appear in the visible observation. Return JSON only. No
    markdown fences. No prose outside the JSON.
    """
).strip()

TASK_BRIEFS: Dict[str, str] = {
    "single_zone_response": (
        "EASY tier. Verify the report stream, allocate the matching unit type "
        "to the true zone before its deadline, then publish a sitrep."
    ),
    "multi_zone_triage": "MEDIUM tier with multiple concurrent incident zones.",
    "cascading_crisis": "HARD tier where incidents stream in mid-episode.",
    "multi_district_coordination": "EXPERT tier with mutual aid and comms degradation.",
}

ACTION_FORMAT_PROMPT = textwrap.dedent(
    """
    Valid action JSON shapes:
    {"type":"verify_report","report_id":"report-1","verification_method":"cross_check","rationale":"..."}
    {"type":"request_recon","zone_id":"zone-1","objective":"...","priority":"normal","report_id":null}
    {"type":"allocate_unit","unit_id":"unit-1","zone_id":"zone-1","task":"rescue","priority":"high","report_ids":["report-1"]}
    {"type":"reroute_unit","unit_id":"unit-1","route":{"route_id":"...","from_zone_id":"...","to_zone_id":"...","status":"open","travel_time_minutes":10,"hazards":[]},"reason":"..."}
    {"type":"issue_evacuation","zone_id":"zone-1","urgency":"critical","message":"...","route_id":null,"destination_shelter_id":null}
    {"type":"open_shelter","shelter":{"shelter_id":"...","zone_id":"...","name":"...","status":"open","capacity_total":100,"capacity_available":50,"supplies":{}},"reason":"..."}
    {"type":"dispatch_supplies","supplies":{"water":100},"destination_zone_id":"zone-1","priority":"high","unit_id":null,"destination_shelter_id":null}
    {"type":"flag_false_alarm","report_id":"report-1","rationale":"...","evidence":["..."]}
    {"type":"publish_sitrep","payload":{"incidents_confirmed":["report-1"],"incidents_resolved":["zone-1"],"unresolved_risks":[],"false_alarms_detected":[],"summary_text":"..."}}
    {"type":"noop","reason":"..."}
    """
).strip()


# --------------------------------------------------------------------------- #
# Training metrics — surfaced into Trackio and the final summary.
# --------------------------------------------------------------------------- #


class TrainingMetrics:
    """Small thread-safe metric bag used by the reward function."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.step_total_reward = 0.0
        self.step_total_count = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.action_type_counts: Dict[str, int] = {}

    def record_invalid(self) -> None:
        with self.lock:
            self.invalid_action_count += 1

    def record_step(self, action: Mapping[str, Any], reward: float) -> None:
        with self.lock:
            self.valid_action_count += 1
            self.step_total_reward += float(reward)
            self.step_total_count += 1
            action_type = str(action.get("type", "unknown"))
            self.action_type_counts[action_type] = (
                self.action_type_counts.get(action_type, 0) + 1
            )

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            total_actions = self.valid_action_count + self.invalid_action_count
            return {
                "average_step_reward": (
                    self.step_total_reward / self.step_total_count
                    if self.step_total_count
                    else 0.0
                ),
                "valid_action_fraction": (
                    self.valid_action_count / total_actions if total_actions else 0.0
                ),
                "valid_action_count": self.valid_action_count,
                "invalid_action_count": self.invalid_action_count,
                "action_type_counts": dict(self.action_type_counts),
            }


# --------------------------------------------------------------------------- #
# HTTP helpers (mirror the smoke test client so behaviour is identical)
# --------------------------------------------------------------------------- #


def post_json(path: str, payload: Mapping[str, Any]) -> dict:
    request = Request(
        f"{ENV_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {ENV_URL}{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Action parsing & sanitization (copied from baseline so training and eval
# share exactly one schema-handling implementation)
# --------------------------------------------------------------------------- #


def parse_action_json(response_text: str) -> dict:
    match = ACTION_JSON_RE.search(response_text or "")
    if not match:
        raise ValueError("model response did not contain a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict) or "type" not in parsed:
        raise ValueError("model response was not an action object")
    return parsed


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"action missing required string field: {key}")
    return value.strip()


def _text_value(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _enum_value(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    return default


def _priority_value(value: Any, default: str) -> str:
    return _enum_value(
        str(value) if value is not None else value,
        {"low", "normal", "high", "critical"},
        default,
    )


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int))]


def sanitize_model_action(parsed: Mapping[str, Any]) -> dict:
    action_type = str(parsed.get("type", ""))
    if not action_type:
        raise ValueError("model response missing type")

    if action_type == "verify_report":
        return {
            "type": "verify_report",
            "report_id": _required_str(parsed, "report_id"),
            "verification_method": _enum_value(
                parsed.get("verification_method"),
                {
                    "cross_check",
                    "contact_source",
                    "field_recon",
                    "sensor_review",
                    "official_confirmation",
                },
                "cross_check",
            ),
            "rationale": _text_value(parsed.get("rationale"), "Verify report."),
        }
    if action_type == "request_recon":
        return {
            "type": "request_recon",
            "zone_id": _required_str(parsed, "zone_id"),
            "objective": _text_value(parsed.get("objective"), "Clarify incident status."),
            "priority": _priority_value(parsed.get("priority"), "normal"),
            "report_id": parsed.get("report_id"),
        }
    if action_type == "allocate_unit":
        return {
            "type": "allocate_unit",
            "unit_id": _required_str(parsed, "unit_id"),
            "zone_id": _required_str(parsed, "zone_id"),
            "task": _enum_value(
                parsed.get("task"),
                {
                    "rescue",
                    "medical",
                    "evacuation",
                    "fire_suppression",
                    "supply_delivery",
                    "recon",
                    "route_clearance",
                },
                "recon",
            ),
            "priority": _priority_value(parsed.get("priority"), "normal"),
            "report_ids": _string_list(parsed.get("report_ids")),
        }
    if action_type == "reroute_unit":
        route = parsed.get("route")
        if not isinstance(route, Mapping):
            raise ValueError("reroute_unit requires route object")
        return {
            "type": "reroute_unit",
            "unit_id": _required_str(parsed, "unit_id"),
            "route": dict(route),
            "reason": _text_value(parsed.get("reason"), "Use safer route."),
        }
    if action_type == "issue_evacuation":
        return {
            "type": "issue_evacuation",
            "zone_id": _required_str(parsed, "zone_id"),
            "urgency": _priority_value(parsed.get("urgency"), "high"),
            "message": _text_value(parsed.get("message"), "Evacuate immediately."),
            "route_id": parsed.get("route_id"),
            "destination_shelter_id": parsed.get("destination_shelter_id"),
        }
    if action_type == "open_shelter":
        shelter = parsed.get("shelter")
        if not isinstance(shelter, Mapping):
            raise ValueError("open_shelter requires shelter object")
        return {
            "type": "open_shelter",
            "shelter": dict(shelter),
            "reason": _text_value(parsed.get("reason"), "Open shelter capacity."),
        }
    if action_type == "dispatch_supplies":
        supplies = parsed.get("supplies")
        if not isinstance(supplies, Mapping) or not supplies:
            raise ValueError("dispatch_supplies requires non-empty supplies")
        sanitized_supplies = {
            str(key): int(value)
            for key, value in supplies.items()
            if isinstance(value, (int, float)) and value > 0
        }
        if not sanitized_supplies:
            raise ValueError("dispatch_supplies requires positive supply amounts")
        return {
            "type": "dispatch_supplies",
            "supplies": sanitized_supplies,
            "destination_zone_id": _required_str(parsed, "destination_zone_id"),
            "priority": _priority_value(parsed.get("priority"), "normal"),
            "unit_id": parsed.get("unit_id"),
            "destination_shelter_id": parsed.get("destination_shelter_id"),
        }
    if action_type == "flag_false_alarm":
        return {
            "type": "flag_false_alarm",
            "report_id": _required_str(parsed, "report_id"),
            "rationale": _text_value(parsed.get("rationale"), "Report is disputed."),
            "evidence": _string_list(parsed.get("evidence")),
        }
    if action_type == "publish_sitrep":
        payload = parsed.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("publish_sitrep requires payload object")
        return {
            "type": "publish_sitrep",
            "payload": {
                "incidents_confirmed": _string_list(payload.get("incidents_confirmed")),
                "incidents_resolved": _string_list(payload.get("incidents_resolved")),
                "unresolved_risks": _string_list(payload.get("unresolved_risks")),
                "false_alarms_detected": _string_list(payload.get("false_alarms_detected")),
                "summary_text": _text_value(
                    payload.get("summary_text"), "Situation report published."
                )[:800],
            },
        }
    if action_type == "noop":
        return {
            "type": "noop",
            "reason": _text_value(parsed.get("reason"), "Waiting for more evidence."),
        }
    raise ValueError(f"unknown action type: {action_type}")


# --------------------------------------------------------------------------- #
# Prompt rendering — same shape as the baseline (no recommended-action block).
# We render fresh for every step request because the env state is mutating.
# --------------------------------------------------------------------------- #


def _compact_observation(obs: Mapping[str, Any]) -> dict:
    return {
        "time_step": obs.get("time_step"),
        "metadata": obs.get("metadata", {}),
        "visible_zones": [
            {
                "zone_id": zone.get("zone_id"),
                "name": zone.get("name"),
                "incident_type": zone.get("incident_type"),
                "severity": zone.get("severity"),
                "population_at_risk": zone.get("population_at_risk"),
                "deadline_steps": zone.get("deadline_steps"),
                "access_status": zone.get("access_status"),
                "district_id": zone.get("district_id"),
                "required_unit_types": sorted(zone.get("required_unit_types") or []),
            }
            for zone in (obs.get("visible_zones") or [])
        ],
        "reports": [
            {
                "report_id": report.get("report_id"),
                "zone_id": report.get("zone_id"),
                "source": report.get("source"),
                "report_type": report.get("report_type"),
                "severity": report.get("severity"),
                "description": (str(report.get("description") or ""))[:200],
                "verified_status": report.get("verified_status"),
                "confidence": report.get("confidence"),
                "reveal_at_step": report.get("reveal_at_step"),
            }
            for report in (obs.get("reports") or [])
        ],
        "resources": [
            {
                "unit_id": unit.get("unit_id"),
                "unit_type": unit.get("unit_type"),
                "status": unit.get("status"),
                "current_zone_id": unit.get("current_zone_id"),
                "capacity": unit.get("capacity"),
                "capabilities": unit.get("capabilities"),
                "fatigue": unit.get("fatigue"),
                "district_id": unit.get("district_id"),
                "mutual_aid_unlock_step": unit.get("mutual_aid_unlock_step"),
            }
            for unit in (obs.get("resources") or [])
        ],
        "incident_log": (obs.get("incident_log") or [])[-6:],
    }


def render_user_prompt(task_id: str, obs: Mapping[str, Any]) -> str:
    brief = TASK_BRIEFS.get(task_id, "(no brief)")
    episode_cap = (obs.get("metadata") or {}).get(
        "episode_cap", TASK_CONFIGS.get(task_id, {}).get("episode_cap", 8)
    )
    return (
        f"Task: {task_id}\n"
        f"Brief: {brief}\n"
        f"Time step: {obs.get('time_step')}\n"
        f"Episode cap: {episode_cap}\n\n"
        f"Action contract:\n{ACTION_FORMAT_PROMPT}\n\n"
        f"Current observation:\n"
        f"{json.dumps(_compact_observation(obs), sort_keys=True)}\n\n"
        "Return exactly one JSON object and nothing else."
    )


def build_prompt_messages(task_id: str, obs: Mapping[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(task_id, obs)},
    ]


# --------------------------------------------------------------------------- #
# Reward function — this is where the env interaction happens. GRPO will call
# this with `completions` (a list of model-generated strings) for each row
# in the dataset batch. We turn each completion into an action, hit /step,
# and return the env's reward.
# --------------------------------------------------------------------------- #

# Metrics created in main() and bound here so the reward function can update it.
_METRICS: Optional[TrainingMetrics] = None


def _completion_text(completion: Any) -> str:
    """Normalize a completion to a string. TRL passes either str or [{role, content}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, Mapping) and "content" in last:
            return str(last.get("content", ""))
    if isinstance(completion, Mapping) and "content" in completion:
        return str(completion.get("content", ""))
    return str(completion)


def crisisops_step_reward(prompts, completions, **kwargs) -> List[float]:
    """One reward per completion: env step reward (action accepted) or penalty."""
    metrics = _METRICS
    seeds = kwargs.get("seed") or [42] * len(completions)
    rewards: List[float] = []
    for index, completion in enumerate(completions):
        text = _completion_text(completion)
        try:
            parsed = parse_action_json(text)
            action = sanitize_model_action(parsed)
        except Exception:
            if metrics is not None:
                metrics.record_invalid()
            rewards.append(-0.5)  # invalid JSON penalty
            continue

        seed = int(seeds[index % len(seeds)])
        episode_id = f"grpo-{seed}-{uuid.uuid4().hex[:8]}"

        try:
            reset_response = post_json(
                "/reset",
                {"task_id": TASK_ID, "seed": seed, "episode_id": episode_id},
            )
        except Exception as exc:
            print(f"[REWARD] /reset error seed={seed} error={exc}", flush=True)
            rewards.append(-0.2)
            continue
        session_id = str(reset_response.get("session_id", ""))
        if not session_id:
            rewards.append(-0.5)
            continue

        try:
            response = post_json(
                "/step", {"session_id": session_id, "action": action}
            )
        except Exception as exc:
            print(f"[REWARD] /step error session={session_id} error={exc}", flush=True)
            rewards.append(-0.2)
            continue

        step_reward = float(response.get("reward") or 0.0)
        reward = step_reward + action_shaping_bonus(action)
        if metrics is not None:
            metrics.record_step(action, reward)

        rewards.append(reward)
    return rewards


def action_shaping_bonus(action: Mapping[str, Any]) -> float:
    """Small bias toward actions that can make progress on the easy task."""
    action_type = str(action.get("type", ""))
    if action_type == "verify_report":
        return 0.05
    if action_type in {"flag_false_alarm", "allocate_unit"}:
        return 0.02
    if action_type in {"request_recon", "noop"}:
        return -0.03
    return 0.0


# --------------------------------------------------------------------------- #
# Trackio metric callback — flush episode-level summary stats every N steps so
# the live dashboard shows the curve we care about (avg episode score).
# --------------------------------------------------------------------------- #


def make_pool_metric_callback():
    from transformers import TrainerCallback

    class PoolMetricCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics = _METRICS
            if metrics is None or logs is None:
                return
            snapshot = metrics.snapshot()
            logs["crisisops/avg_step_reward"] = snapshot["average_step_reward"]
            logs["crisisops/valid_action_fraction"] = snapshot["valid_action_fraction"]
            logs["crisisops/valid_action_count"] = snapshot["valid_action_count"]

    return PoolMetricCallback


def patch_text_only_unsloth_grpo_trainer(trainer: Any) -> None:
    """Patch text-only compatibility gaps in Unsloth's generated GRPO trainer."""
    for attr in (
        "image_token",
        "image_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        if not hasattr(trainer, attr):
            setattr(trainer, attr, None)
    processing_class = getattr(trainer, "processing_class", None)
    for attr in ("pad_token", "pad_token_id", "eos_token", "eos_token_id"):
        if not hasattr(trainer, attr) and hasattr(processing_class, attr):
            setattr(trainer, attr, getattr(processing_class, attr))

    def truncate_with_protected_tokens(input_ids, attention_mask, max_length, protected):
        del protected
        if max_length is None:
            return input_ids, attention_mask
        if input_ids.shape[-1] <= max_length:
            return input_ids, attention_mask
        return input_ids[..., -max_length:], attention_mask[..., -max_length:]

    patched_scopes = 0

    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("UnslothGRPOTrainer"):
            module.__dict__.setdefault(
                "truncate_with_protected_tokens", truncate_with_protected_tokens
            )
            patched_scopes += 1

    method = getattr(trainer, "_generate_and_score_completions", None)
    candidates = [method, getattr(method, "__func__", None)]
    for candidate in candidates:
        while candidate is not None:
            globals_dict = getattr(candidate, "__globals__", None)
            if isinstance(globals_dict, dict):
                globals_dict["truncate_with_protected_tokens"] = (
                    truncate_with_protected_tokens
                )
                patched_scopes += 1
            candidate = getattr(candidate, "__wrapped__", None)

    print(
        f"[TRAIN] patched_text_only_unsloth_grpo_trainer scopes={patched_scopes}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# CUDA / GPU sanity (carry over from the smoke test so HF Job logs are useful)
# --------------------------------------------------------------------------- #


def log_gpu_preflight() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], check=False, capture_output=True, text=True
        )
        output = (result.stdout or result.stderr or "").strip()
        if output:
            print(f"[TRAIN] nvidia_smi={output}", flush=True)
    except Exception as exc:
        print(f"[TRAIN] nvidia_smi_unavailable={exc}", flush=True)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or result.stderr or "").strip()
        if output:
            print(f"[TRAIN] nvidia_smi_detail={output}", flush=True)
    except Exception as exc:
        print(f"[TRAIN] nvidia_smi_detail_unavailable={exc}", flush=True)


def wait_for_cuda_runtime() -> None:
    """Wait for CUDA in child processes so failed probes do not poison training."""
    retries = int(os.getenv("CUDA_WAIT_RETRIES", "60"))
    sleep_seconds = int(os.getenv("CUDA_WAIT_SLEEP_SECONDS", "10"))
    probe_timeout = int(os.getenv("CUDA_PROBE_TIMEOUT", "180"))
    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os, sys, torch; "
                        "print('torch_version=' + torch.__version__); "
                        "print('cuda_visible_devices=' + str(os.environ.get('CUDA_VISIBLE_DEVICES'))); "
                        "sys.stdout.flush(); "
                        "torch.cuda.init(); "
                        "print('device_count=' + str(torch.cuda.device_count())); "
                        "print('device_name=' + torch.cuda.get_device_name(0))"
                    ),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=probe_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            partial_output = ""
            if exc.output:
                partial_output += exc.output.decode("utf-8", errors="replace") if isinstance(exc.output, bytes) else exc.output
            if exc.stderr:
                partial_output += exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
            last_error = f"probe timed out after {probe_timeout}s"
            if partial_output.strip():
                last_error += f" partial_output={partial_output.strip()}"
            print(
                f"[TRAIN] cuda_not_ready attempt={attempt}/{retries} "
                f"sleep_seconds={sleep_seconds} last_error={last_error}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(sleep_seconds)
            continue
        if probe.returncode == 0:
            output = (probe.stdout or "").strip()
            print(f"[TRAIN] cuda_ready probe={output}", flush=True)
            return
        last_error = ((probe.stderr or "") + (probe.stdout or "")).strip()
        print(
            f"[TRAIN] cuda_not_ready attempt={attempt}/{retries} "
            f"sleep_seconds={sleep_seconds} last_error={last_error or 'none'}",
            flush=True,
        )
        if attempt < retries:
            time.sleep(sleep_seconds)
    hint = (
        " Error 802/system not yet initialized means nvidia-smi can see the "
        "GPU but the CUDA runtime cannot initialize compute in this job "
        "container yet. Increase CUDA_WAIT_RETRIES if it is transient; if it "
        "persists for the full wait, the failure is below the GRPO/Unsloth "
        "training code."
    )
    raise RuntimeError(
        "CUDA did not become ready inside the HF Job. "
        f"Last error: {last_error or 'torch.cuda probe failed'}." + hint
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    global _METRICS

    if TASK_ID not in TASK_CONFIGS:
        raise ValueError(f"Unknown TASK_ID={TASK_ID!r}")

    print(
        f"[TRAIN] env_url={ENV_URL} task_id={TASK_ID} model_id={MODEL_ID} "
        f"max_steps={GRPO_MAX_STEPS} num_generations={GRPO_NUM_GENERATIONS} "
        f"per_device_batch={GRPO_PER_DEVICE_BATCH} grad_accum={GRPO_GRAD_ACCUM}",
        flush=True,
    )

    log_gpu_preflight()
    wait_for_cuda_runtime()

    import torch

    torch.cuda.init()

    print(
        f"[TRAIN] python={platform.python_version()} platform={platform.platform()}",
        flush=True,
    )
    print(
        f"[TRAIN] cuda_device_count={torch.cuda.device_count()} "
        f"device_name={torch.cuda.get_device_name(0)} "
        f"cuda_capability={torch.cuda.get_device_capability(0)}",
        flush=True,
    )

    # ---- Load model + LoRA via unsloth -----------------------------------
    from unsloth import FastLanguageModel

    print(
        f"[TRAIN] loading model={MODEL_ID} max_seq_length={MAX_SEQ_LENGTH}",
        flush=True,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=GRPO_LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=GRPO_LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[TRAIN] model loaded and LoRA adapter attached", flush=True)

    # ---- Build prompt pool from deterministic resets ---------------------
    seeds = [42 + offset for offset in range(SEED_POOL_SIZE)]
    _METRICS = TrainingMetrics()
    prompt_observations: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        response = post_json(
            "/reset",
            {
                "task_id": TASK_ID,
                "seed": seed,
                "episode_id": f"prompt-{seed}-{uuid.uuid4().hex[:8]}",
            },
        )
        prompt_observations[seed] = response.get("observation") or {}
    print(f"[TRAIN] prompt pool warmed seeds={seeds}", flush=True)

    # ---- Build dataset -----------------------------------------------------
    # Each row carries the seed that produced its prompt. The reward function
    # resets that same seed before applying the completion, so prompt and
    # scored environment state stay aligned.
    from datasets import Dataset

    def make_row(index: int) -> Dict[str, Any]:
        seed = seeds[index % len(seeds)]
        messages = build_prompt_messages(TASK_ID, prompt_observations[seed])
        # GRPOTrainer accepts prompts as either string or list-of-messages;
        # list-of-messages avoids manual chat template formatting.
        return {
            "prompt": messages,
            "seed": seed,
            "task_id": TASK_ID,
        }

    rows = [make_row(i) for i in range(DATASET_ROWS)]
    train_dataset = Dataset.from_list(rows)
    print(f"[TRAIN] built dataset rows={len(train_dataset)}", flush=True)

    # ---- GRPO config -------------------------------------------------------
    from trl import GRPOConfig, GRPOTrainer

    if TRACKIO_SPACE_ID:
        os.environ["TRACKIO_SPACE_ID"] = TRACKIO_SPACE_ID
        os.environ.setdefault("TRACKIO_PROJECT", "crisisops-grpo")

    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=GRPO_MAX_STEPS,
        learning_rate=GRPO_LEARNING_RATE,
        beta=GRPO_BETA,
        optim="paged_adamw_8bit",
        max_grad_norm=GRPO_MAX_GRAD_NORM,
        per_device_train_batch_size=GRPO_PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRPO_GRAD_ACCUM,
        num_generations=GRPO_NUM_GENERATIONS,
        warmup_steps=GRPO_WARMUP_STEPS,
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH,
        max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
        temperature=GRPO_TEMPERATURE,
        top_p=GRPO_TOP_P,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=GRPO_LOGGING_STEPS,
        save_steps=GRPO_SAVE_STEPS,
        save_total_limit=3,
        push_to_hub=bool(HF_REPO_ID),
        hub_token=HF_TOKEN,
        hub_model_id=HF_REPO_ID,
        hub_strategy="every_save" if HF_REPO_ID else "end",
        report_to="trackio" if TRACKIO_SPACE_ID else "none",
        run_name=RUN_NAME,
        bf16=True,
    )
    grpo_config = GRPOConfig(**grpo_kwargs)

    PoolMetricCallback = make_pool_metric_callback()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[crisisops_step_reward],
        args=grpo_config,
        train_dataset=train_dataset,
        callbacks=[PoolMetricCallback()],
    )
    patch_text_only_unsloth_grpo_trainer(trainer)

    # ---- Train -------------------------------------------------------------
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    total_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)
    print(
        f"[TRAIN] memory_before reserved_gb={start_mem} total_gb={total_mem}",
        flush=True,
    )

    train_start = time.time()
    trainer.train()
    train_end = time.time()

    peak_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(
        f"[TRAIN] memory_after peak_gb={peak_mem} duration_minutes={(train_end - train_start) / 60:.2f}",
        flush=True,
    )

    # ---- Save adapter ------------------------------------------------------
    save_dir = OUTPUT_DIR
    trainer.save_model(save_dir)
    if HF_REPO_ID:
        try:
            trainer.push_to_hub()
            print(f"[TRAIN] pushed adapter to {HF_REPO_ID}", flush=True)
        except Exception as exc:
            print(f"[TRAIN] push_to_hub failed error={exc}", flush=True)

    # ---- Final pool snapshot ---------------------------------------------
    final = _METRICS.snapshot()
    summary_path = os.path.join(save_dir, "crisisops_training_summary.json")
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "task_id": TASK_ID,
                    "env_url": ENV_URL,
                    "model_id": MODEL_ID,
                    "grpo_max_steps": GRPO_MAX_STEPS,
                    "seeds": seeds,
                    "training_minutes": (train_end - train_start) / 60,
                    **final,
                },
                handle,
                indent=2,
            )
        print(f"[TRAIN] wrote summary to {summary_path}", flush=True)
    except Exception as exc:
        print(f"[TRAIN] summary write failed error={exc}", flush=True)

    print(
        f"[TRAIN] DONE avg_step_reward={final['average_step_reward']:.3f} "
        f"valid_action_fraction={final['valid_action_fraction']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
