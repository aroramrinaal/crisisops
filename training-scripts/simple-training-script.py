"""
GRPO training run for CrisisOps + Unsloth on a single H200.

Step-level training: each GRPO "step request" is one model action against the
deployed CrisisOps HF Space. The model writes a JSON action, the env applies
it, and the env's per-step reward becomes the GRPO reward. Across a training
step we run `per_device_train_batch_size * gradient_accumulation_steps *
num_generations` of these action requests, GRPO normalizes within each group
and updates the LoRA adapter.

Why step-level instead of episode-level:
- Plays nicely with TRL's default GRPOTrainer contract (one prompt, one
  completion, one reward) instead of the experimental rollout_func path.
- Many more reward observations per training hour, so the curve is smoother.
- The per-step reward in `server/reward.py` already encodes trajectory-level
  signals (deadline misses, resolved-before-deadline bonuses), so episode-
  level credit assignment is not strictly needed for a hackathon-grade
  improvement curve.

The script:
1. Loads Qwen2.5-Coder-3B-Instruct in 4-bit via unsloth + LoRA r=16
2. Maintains a small pool of live CrisisOps episodes keyed by (seed, episode_id)
3. Builds a HF Dataset of N step-request rows. Each row points at one episode
   in the pool. When a reward function pulls the row, it renders a fresh prompt
   from the env's current observation, calls /step with the model action, and
   returns the env reward.
4. Logs to trackio so the loss + reward curve are visible from a HF Space.

Usage:

hf jobs uv run \
  --flavor h200 \
  --timeout 4h \
  --secrets HF_TOKEN \
  --with "git+https://github.com/huggingface/trl.git" \
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
import textwrap
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# --------------------------------------------------------------------------- #
# Config (env-var driven so HF Jobs can override without code changes)
# --------------------------------------------------------------------------- #

ENV_URL = os.getenv("ENV_URL", "https://mrinaalarora-crisisops.hf.space").rstrip("/")
MODEL_ID = os.getenv("MODEL_ID", "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit").strip()
TASK_ID = os.getenv("TASK_ID", "single_zone_response").strip()

# Training scale. 100 GRPO steps with the sizing below = 100 weight updates,
# 100 * 4 (grad_accum) * 2 (per-device batch) * 4 (num_generations) = 3200
# action requests against the live HF Space. On an H200 with a 3B 4-bit model
# this should land somewhere in the 30-45 minute range.
GRPO_MAX_STEPS = int(os.getenv("GRPO_MAX_STEPS", "100"))
GRPO_NUM_GENERATIONS = int(os.getenv("GRPO_NUM_GENERATIONS", "4"))
GRPO_PER_DEVICE_BATCH = int(os.getenv("GRPO_PER_DEVICE_BATCH", "2"))
GRPO_GRAD_ACCUM = int(os.getenv("GRPO_GRAD_ACCUM", "4"))
GRPO_LEARNING_RATE = float(os.getenv("GRPO_LEARNING_RATE", "5e-6"))
GRPO_BETA = float(os.getenv("GRPO_BETA", "0.04"))
GRPO_TEMPERATURE = float(os.getenv("GRPO_TEMPERATURE", "0.9"))
GRPO_TOP_P = float(os.getenv("GRPO_TOP_P", "0.95"))
GRPO_MAX_PROMPT_LENGTH = int(os.getenv("GRPO_MAX_PROMPT_LENGTH", "3072"))
GRPO_MAX_COMPLETION_LENGTH = int(os.getenv("GRPO_MAX_COMPLETION_LENGTH", "512"))
GRPO_WARMUP_STEPS = int(os.getenv("GRPO_WARMUP_STEPS", "5"))
GRPO_LOGGING_STEPS = int(os.getenv("GRPO_LOGGING_STEPS", "1"))
GRPO_SAVE_STEPS = int(os.getenv("GRPO_SAVE_STEPS", "25"))
GRPO_LORA_RANK = int(os.getenv("GRPO_LORA_RANK", "16"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "4096"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "crisisops-grpo-easy-lora").strip()
HF_REPO_ID = os.getenv("HF_REPO_ID", "").strip() or None
TRACKIO_SPACE_ID = os.getenv("TRACKIO_SPACE_ID", "").strip() or None

# Episode pool. We rotate through this many fixed seeds to keep the reward
# curve low-variance (recommended over fully random seeds for a short run).
SEED_POOL_SIZE = int(os.getenv("SEED_POOL_SIZE", "5"))
EPISODE_TRAINING_BUDGET = int(os.getenv("EPISODE_TRAINING_BUDGET", "16"))

# How many step-request rows to put in the dataset. GRPOTrainer iterates over
# these; with num_train_epochs=1 and max_steps capped, we just need enough to
# cover max_steps * per_device_batch * grad_accum. We oversize so the trainer
# never runs out before max_steps.
DATASET_ROWS = int(
    os.getenv(
        "DATASET_ROWS",
        str(max(2048, GRPO_MAX_STEPS * GRPO_PER_DEVICE_BATCH * GRPO_GRAD_ACCUM * 4)),
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
# Episode pool — keeps live CrisisOps sessions across training steps
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeState:
    """One live CrisisOps session that GRPO step requests pull from."""

    seed: int
    session_id: str = ""
    observation: Dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    episode_index: int = 0
    done: bool = True  # start in 'done' so first request triggers a reset

    def key(self) -> str:
        return f"seed-{self.seed}"


class EpisodePool:
    """Thread-safe pool of CrisisOps episodes keyed by seed."""

    def __init__(self, seeds: List[int], task_id: str):
        self.task_id = task_id
        self.lock = threading.RLock()
        self.episodes: Dict[int, EpisodeState] = {seed: EpisodeState(seed=seed) for seed in seeds}
        # Round-robin pointer used when a reward function asks for a step
        # without specifying which seed to use.
        self._round_robin = 0
        self._seeds = list(seeds)
        # Counters surfaced into trackio
        self.metrics = {
            "episodes_completed": 0,
            "episode_total_score": 0.0,
            "episode_score_count": 0,
            "step_total_reward": 0.0,
            "step_total_count": 0,
            "valid_action_count": 0,
            "invalid_action_count": 0,
            "action_type_counts": {},
        }

    def reset_episode(self, episode: EpisodeState) -> EpisodeState:
        episode.episode_index += 1
        episode.step_count = 0
        episode.done = False
        episode.session_id = ""
        episode_id = f"train-{episode.seed}-{episode.episode_index}-{uuid.uuid4().hex[:8]}"
        try:
            response = post_json(
                "/reset",
                {"task_id": self.task_id, "seed": episode.seed, "episode_id": episode_id},
            )
        except Exception as exc:
            print(f"[POOL] reset failed seed={episode.seed} error={exc}", flush=True)
            episode.done = True
            return episode
        episode.session_id = str(response.get("session_id", episode_id))
        episode.observation = response.get("observation") or {}
        return episode

    def ensure_live(self, episode: EpisodeState) -> EpisodeState:
        if episode.done or not episode.session_id:
            self.reset_episode(episode)
        return episode

    def next_episode(self) -> EpisodeState:
        """Round-robin selection of an episode for the next step request."""
        with self.lock:
            seed = self._seeds[self._round_robin % len(self._seeds)]
            self._round_robin += 1
            episode = self.episodes[seed]
            self.ensure_live(episode)
            return episode

    def apply_step_result(
        self,
        episode: EpisodeState,
        action: Mapping[str, Any],
        response: Mapping[str, Any],
        reward: float,
    ) -> Tuple[Optional[float], bool]:
        """Update episode state from a /step response. Returns (terminal_score, done)."""
        next_obs = response.get("observation") or {}
        done = bool(response.get("done", False) or next_obs.get("done", False))
        with self.lock:
            episode.observation = next_obs
            episode.step_count = int(next_obs.get("time_step", episode.step_count + 1))
            episode.done = done
            self.metrics["step_total_reward"] += float(reward)
            self.metrics["step_total_count"] += 1
            atype = str(action.get("type", "unknown"))
            self.metrics["action_type_counts"][atype] = (
                self.metrics["action_type_counts"].get(atype, 0) + 1
            )
            terminal_score = None
            if done:
                metadata = next_obs.get("metadata") or {}
                if "terminal_score" in metadata:
                    terminal_score = float(metadata["terminal_score"])
                    self.metrics["episode_total_score"] += terminal_score
                    self.metrics["episode_score_count"] += 1
                self.metrics["episodes_completed"] += 1
        return terminal_score, done

    def record_invalid(self) -> None:
        with self.lock:
            self.metrics["invalid_action_count"] += 1

    def record_valid(self) -> None:
        with self.lock:
            self.metrics["valid_action_count"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            avg_step_reward = (
                self.metrics["step_total_reward"] / self.metrics["step_total_count"]
                if self.metrics["step_total_count"]
                else 0.0
            )
            avg_episode_score = (
                self.metrics["episode_total_score"] / self.metrics["episode_score_count"]
                if self.metrics["episode_score_count"]
                else 0.0
            )
            total_actions = (
                self.metrics["valid_action_count"] + self.metrics["invalid_action_count"]
            )
            valid_fraction = (
                self.metrics["valid_action_count"] / total_actions
                if total_actions
                else 0.0
            )
            return {
                "episodes_completed": self.metrics["episodes_completed"],
                "average_episode_score": avg_episode_score,
                "average_step_reward": avg_step_reward,
                "valid_action_fraction": valid_fraction,
                "action_type_counts": dict(self.metrics["action_type_counts"]),
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

# Pool created in main() and bound here so the reward function can see it.
_POOL: Optional[EpisodePool] = None
_POOL_LOCK = threading.RLock()


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
    pool = _POOL
    rewards: List[float] = []
    for completion in completions:
        text = _completion_text(completion)
        try:
            parsed = parse_action_json(text)
            action = sanitize_model_action(parsed)
        except Exception:
            if pool is not None:
                pool.record_invalid()
            rewards.append(-0.5)  # invalid JSON penalty
            continue

        if pool is None:
            rewards.append(0.0)
            continue
        pool.record_valid()

        episode = pool.next_episode()
        if not episode.session_id:
            rewards.append(-0.5)
            continue

        try:
            response = post_json(
                "/step",
                {"session_id": episode.session_id, "action": action},
            )
        except Exception as exc:
            print(f"[REWARD] /step error session={episode.session_id} error={exc}", flush=True)
            with pool.lock:
                episode.done = True
            rewards.append(-0.2)
            continue

        step_reward = float(response.get("reward") or 0.0)
        terminal_score, done = pool.apply_step_result(episode, action, response, step_reward)

        # Combined reward: step reward always, + terminal_score bonus when the
        # episode actually finishes via publish_sitrep (or hits cap with one).
        # The terminal score is in [0.01, 0.99] from EasyGrader, so this gives
        # a strong end-of-episode signal that the per-step reward alone lacks.
        reward = step_reward
        if done and terminal_score is not None:
            reward += terminal_score

        rewards.append(reward)
    return rewards


# --------------------------------------------------------------------------- #
# Trackio metric callback — flush episode-level summary stats every N steps so
# the live dashboard shows the curve we care about (avg episode score).
# --------------------------------------------------------------------------- #


def make_pool_metric_callback():
    from transformers import TrainerCallback

    class PoolMetricCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            pool = _POOL
            if pool is None or logs is None:
                return
            snapshot = pool.snapshot()
            logs["crisisops/avg_episode_score"] = snapshot["average_episode_score"]
            logs["crisisops/avg_step_reward"] = snapshot["average_step_reward"]
            logs["crisisops/episodes_completed"] = snapshot["episodes_completed"]
            logs["crisisops/valid_action_fraction"] = snapshot["valid_action_fraction"]

    return PoolMetricCallback


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


def wait_for_cuda(torch, retries: int = 6, sleep_seconds: int = 5) -> None:
    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                if not torch.cuda.is_initialized():
                    torch.cuda.init()
                return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        print(
            f"[TRAIN] cuda_not_ready attempt={attempt}/{retries} "
            f"sleep_seconds={sleep_seconds} last_error={last_error or 'none'}",
            flush=True,
        )
        time.sleep(sleep_seconds)
    raise RuntimeError(
        "CUDA did not become ready inside the HF Job. "
        f"Last error: {last_error or 'torch.cuda.is_available() returned false'}"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    global _POOL

    if TASK_ID not in TASK_CONFIGS:
        raise ValueError(f"Unknown TASK_ID={TASK_ID!r}")

    print(
        f"[TRAIN] env_url={ENV_URL} task_id={TASK_ID} model_id={MODEL_ID} "
        f"max_steps={GRPO_MAX_STEPS} num_generations={GRPO_NUM_GENERATIONS} "
        f"per_device_batch={GRPO_PER_DEVICE_BATCH} grad_accum={GRPO_GRAD_ACCUM}",
        flush=True,
    )

    import torch

    log_gpu_preflight()
    wait_for_cuda(torch)

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

    # ---- Build episode pool with rotating fixed seeds --------------------
    seeds = [42 + offset for offset in range(SEED_POOL_SIZE)]
    _POOL = EpisodePool(seeds=seeds, task_id=TASK_ID)
    # Pre-warm one episode per seed so the first GRPO step doesn't hit a
    # cold start for every seed at once.
    for seed in seeds:
        _POOL.ensure_live(_POOL.episodes[seed])
    print(f"[TRAIN] episode pool warmed seeds={seeds}", flush=True)

    # ---- Build dataset -----------------------------------------------------
    # Each row carries the seed it should pull from and a per-row prompt
    # rendered from that episode's CURRENT observation. The rendered prompt
    # ages quickly as the episode advances, but GRPOTrainer regenerates the
    # tokenized prompt on access so this is fine for warmup. The reward
    # function does the real env interaction at completion time.
    from datasets import Dataset

    def make_row(index: int) -> Dict[str, Any]:
        seed = seeds[index % len(seeds)]
        episode = _POOL.episodes[seed]
        _POOL.ensure_live(episode)
        messages = build_prompt_messages(TASK_ID, episode.observation)
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

    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=GRPO_MAX_STEPS,
        learning_rate=GRPO_LEARNING_RATE,
        beta=GRPO_BETA,
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
        hub_model_id=HF_REPO_ID,
        hub_strategy="every_save" if HF_REPO_ID else "end",
        report_to="trackio" if TRACKIO_SPACE_ID else "none",
        bf16=True,
    )
    if TRACKIO_SPACE_ID:
        grpo_kwargs["trackio_space_id"] = TRACKIO_SPACE_ID
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
    final = _POOL.snapshot()
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
        f"[TRAIN] DONE episodes_completed={final['episodes_completed']} "
        f"avg_episode_score={final['average_episode_score']:.3f} "
        f"avg_step_reward={final['average_step_reward']:.3f} "
        f"valid_action_fraction={final['valid_action_fraction']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
