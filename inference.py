"""
Baseline inference script for the CrisisOps OpenEnv environment.

The script is intentionally conservative: it uses a deterministic incident
command policy as the safe fallback, and can ask an OpenAI-compatible model to
return the same JSON action shape when API credentials are available.

Required hackathon-style env vars:
    ENV_URL        Base URL of the CrisisOps FastAPI server.
    API_BASE_URL   OpenAI-compatible API base URL.
    MODEL_NAME     Model identifier.
    HF_TOKEN       Hugging Face/API token. API_KEY is also accepted.

Optional env vars:
    TASK_ID        One CrisisOps task id to run.
    TASK_NAME      Alias for TASK_ID. Tier names easy/medium/hard/expert work.
    SEED           Optional scenario seed passed to /reset.
    USE_LLM        Set to 0 to skip model calls and use the deterministic policy.
    MAX_STEPS      Per-episode safety cap. Defaults to the task episode cap.

STDOUT format:
    [START] task=<task_id> env=crisisops policy=<deterministic|llm> model=<model_name|none>
    [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import textwrap
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - lets the deterministic policy still run.
    OpenAI = None  # type: ignore[assignment]


ENV_NAME = "crisisops"
ENV_URL = os.getenv("ENV_URL", "https://mrinaalarora-crisisops.hf.space").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
USE_LLM = os.getenv("USE_LLM", "1").strip().lower() not in {"0", "false", "no", "off"}
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.35"))

TASK_TIERS: Dict[str, str] = {
    "single_zone_response": "easy",
    "multi_zone_triage": "medium",
    "cascading_crisis": "hard",
    "multi_district_coordination": "expert",
}

TASK_BY_TIER = {tier: task_id for task_id, tier in TASK_TIERS.items()}

TASK_CONFIGS: Dict[str, Dict[str, int]] = {
    "single_zone_response": {
        "episode_cap": 8,
        "initial_reports": 3,
        "expected_reports": 3,
        "stream_done_step": 0,
    },
    "multi_zone_triage": {
        "episode_cap": 15,
        "initial_reports": 6,
        "expected_reports": 6,
        "stream_done_step": 0,
    },
    "cascading_crisis": {
        "episode_cap": 25,
        "initial_reports": 10,
        "expected_reports": 10,
        "stream_done_step": 12,
    },
    "multi_district_coordination": {
        "episode_cap": 40,
        "initial_reports": 15,
        "expected_reports": 16,
        "stream_done_step": 20,
    },
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

ACTION_JSON_RE = re.compile(r"\{[\s\S]*\}")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an emergency operations commander for CrisisOps. Return exactly one
    JSON object for the next action. Do not include markdown or explanation.

    Prefer the provided deterministic candidate unless the observation clearly
    proves it is invalid. Valid action types are:
    verify_report, request_recon, allocate_unit, reroute_unit, issue_evacuation,
    open_shelter, dispatch_supplies, flag_false_alarm, publish_sitrep, noop.
    """
).strip()


@dataclass
class EpisodeMemory:
    verified_report_ids: set[str] = field(default_factory=set)
    true_report_ids: set[str] = field(default_factory=set)
    false_report_ids: set[str] = field(default_factory=set)
    flagged_false_ids: set[str] = field(default_factory=set)
    allocated_unit_ids: set[str] = field(default_factory=set)
    allocated_types_by_zone: Dict[str, set[str]] = field(default_factory=dict)
    resolved_zone_ids: set[str] = field(default_factory=set)
    known_reports: Dict[str, dict] = field(default_factory=dict)
    known_zones: Dict[str, dict] = field(default_factory=dict)
    known_units: Dict[str, dict] = field(default_factory=dict)

    def update_from_observation(self, obs: Mapping[str, Any]) -> None:
        for zone in obs.get("visible_zones", []) or []:
            zone_id = str(zone.get("zone_id", ""))
            if zone_id:
                self.known_zones[zone_id] = dict(zone)

        for unit in obs.get("resources", []) or []:
            unit_id = str(unit.get("unit_id", ""))
            if not unit_id:
                continue
            self.known_units[unit_id] = dict(unit)
            zone_id = unit.get("current_zone_id")
            unit_type = unit.get("unit_type")
            if unit.get("status") in {"assigned", "en_route"} and zone_id and unit_type:
                self.allocated_unit_ids.add(unit_id)
                self.allocated_types_by_zone.setdefault(str(zone_id), set()).add(
                    str(unit_type)
                )
                if _unit_type_matches_zone(str(unit_type), self.known_zones.get(str(zone_id), {})):
                    self.resolved_zone_ids.add(str(zone_id))

        for report in obs.get("reports", []) or []:
            report_id = str(report.get("report_id", ""))
            if not report_id:
                continue
            self.known_reports[report_id] = dict(report)
            status = report.get("verified_status")
            confidence = report.get("confidence")
            if status == "verified":
                self.verified_report_ids.add(report_id)
                self.true_report_ids.add(report_id)
            elif status in {"disputed", "false_alarm"}:
                self.verified_report_ids.add(report_id)
                self.false_report_ids.add(report_id)
            elif confidence == "sensor_confirmed":
                self.true_report_ids.add(report_id)

    def remember_action(self, action: Mapping[str, Any]) -> None:
        action_type = action.get("type")
        if action_type == "verify_report":
            report_id = str(action.get("report_id", ""))
            if report_id:
                self.verified_report_ids.add(report_id)
        elif action_type == "flag_false_alarm":
            report_id = str(action.get("report_id", ""))
            if report_id:
                self.flagged_false_ids.add(report_id)
                self.false_report_ids.add(report_id)
        elif action_type == "allocate_unit":
            unit_id = str(action.get("unit_id", ""))
            zone_id = str(action.get("zone_id", ""))
            unit = self.known_units.get(unit_id, {})
            unit_type = str(unit.get("unit_type", ""))
            if unit_id:
                self.allocated_unit_ids.add(unit_id)
            if zone_id and unit_type:
                self.allocated_types_by_zone.setdefault(zone_id, set()).add(unit_type)
                if _unit_type_matches_zone(unit_type, self.known_zones.get(zone_id, {})):
                    self.resolved_zone_ids.add(zone_id)


def log_start(task: str, client: Any) -> None:
    if client is None:
        print(
            f"[START] task={task} env={ENV_NAME} policy=deterministic model=none",
            flush=True,
        )
        return
    print(
        f"[START] task={task} env={ENV_NAME} policy=llm model={MODEL_NAME}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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


def env_reset(task_id: str) -> dict:
    payload: Dict[str, Any] = {"task_id": task_id}
    if os.getenv("SEED"):
        payload["seed"] = int(os.environ["SEED"])
    return post_json("/reset", payload)


def env_step(action: Mapping[str, Any]) -> dict:
    return post_json("/step", {"action": dict(action)})


def make_policy_action(task_id: str, obs: Mapping[str, Any], memory: EpisodeMemory) -> dict:
    memory.update_from_observation(obs)
    config = TASK_CONFIGS[task_id]
    time_step = int(obs.get("time_step", 0) or 0)
    episode_cap = int((obs.get("metadata") or {}).get("episode_cap", config["episode_cap"]))
    remaining = max(0, episode_cap - time_step)

    if remaining <= 1:
        return build_sitrep(memory)

    urgent_allocation = next_allocation_action(memory, critical_only=True)
    if urgent_allocation is not None:
        return urgent_allocation

    false_action = next_false_alarm_action(memory)
    if false_action is not None and remaining > 2:
        return false_action

    verification = next_verification_action(memory)
    if verification is not None and remaining > 2:
        return verification

    allocation = next_allocation_action(memory, critical_only=False)
    if allocation is not None:
        return allocation

    if should_publish(task_id, obs, memory, remaining):
        return build_sitrep(memory)

    return {
        "type": "noop",
        "reason": "Wait for additional streamed reports or mutual-aid availability.",
    }


def next_false_alarm_action(memory: EpisodeMemory) -> Optional[dict]:
    for report_id in sorted(memory.false_report_ids):
        if report_id in memory.flagged_false_ids:
            continue
        return {
            "type": "flag_false_alarm",
            "report_id": report_id,
            "rationale": "Verification disputed this report, so mark it as a false alarm.",
            "evidence": ["verification_status_disputed"],
        }
    return None


def next_verification_action(memory: EpisodeMemory) -> Optional[dict]:
    reports = sorted(
        memory.known_reports.values(),
        key=lambda report: (
            _zone_sort_key(memory.known_zones.get(str(report.get("zone_id", "")), {})),
            int(report.get("reveal_at_step", report.get("time_step", 0)) or 0),
            str(report.get("report_id", "")),
        ),
    )
    for report in reports:
        report_id = str(report.get("report_id", ""))
        if not report_id or report_id in memory.verified_report_ids:
            continue
        if report_id in memory.true_report_ids and report.get("confidence") == "sensor_confirmed":
            continue
        if report.get("verified_status") in {"verified", "disputed", "false_alarm"}:
            continue
        return {
            "type": "verify_report",
            "report_id": report_id,
            "verification_method": _verification_method(report),
            "rationale": "Confirm report truth before committing scarce response units.",
        }
    return None


def next_allocation_action(
    memory: EpisodeMemory,
    critical_only: bool = False,
) -> Optional[dict]:
    zones = sorted(memory.known_zones.values(), key=_zone_sort_key)
    true_zone_ids = _true_zone_ids(memory)
    for zone in zones:
        zone_id = str(zone.get("zone_id", ""))
        if not zone_id or zone_id not in true_zone_ids:
            continue
        if critical_only and int(zone.get("severity", 0) or 0) < 4:
            continue
        if zone.get("access_status") == "blocked":
            continue
        needed_types = _required_unit_types(zone)
        allocated_types = memory.allocated_types_by_zone.get(zone_id, set())
        for unit_type in sorted(needed_types - allocated_types):
            unit = _best_available_unit(memory, unit_type, zone)
            if unit is None:
                continue
            report_ids = _supporting_true_reports(memory, zone_id)
            return {
                "type": "allocate_unit",
                "unit_id": unit["unit_id"],
                "zone_id": zone_id,
                "task": TASK_BY_UNIT_TYPE.get(unit_type, "recon"),
                "priority": _priority_for_zone(zone),
                "report_ids": report_ids,
            }
    return None


def should_publish(
    task_id: str,
    obs: Mapping[str, Any],
    memory: EpisodeMemory,
    remaining: int,
) -> bool:
    if remaining <= 2:
        return True

    config = TASK_CONFIGS[task_id]
    time_step = int(obs.get("time_step", 0) or 0)
    visible_count = len(memory.known_reports)
    stream_complete = (
        visible_count >= config["expected_reports"]
        or time_step >= config["stream_done_step"]
    )
    no_more_known_work = (
        next_false_alarm_action(memory) is None
        and next_allocation_action(memory, critical_only=False) is None
        and next_verification_action(memory) is None
    )
    return stream_complete and no_more_known_work


def build_sitrep(memory: EpisodeMemory) -> dict:
    confirmed = sorted(memory.true_report_ids)
    false_alarms = sorted(memory.false_report_ids | memory.flagged_false_ids)
    true_zones = _true_zone_ids(memory)
    resolved = sorted(memory.resolved_zone_ids & true_zones)
    unresolved = sorted(true_zones - set(resolved))
    summary = (
        "Verified reports were triaged by deadline, severity, and population at risk. "
        f"Confirmed {len(confirmed)} incident reports, resolved {len(resolved)} zones, "
        f"and identified {len(false_alarms)} false alarms."
    )
    return {
        "type": "publish_sitrep",
        "payload": {
            "incidents_confirmed": confirmed,
            "incidents_resolved": resolved,
            "unresolved_risks": unresolved,
            "false_alarms_detected": false_alarms,
            "summary_text": summary,
        },
    }


def choose_action(
    client: Any,
    task_id: str,
    obs: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict:
    if client is None or not USE_LLM:
        return dict(candidate)

    prompt = textwrap.dedent(
        f"""
        Task: {task_id}
        Time step: {obs.get("time_step")}
        Candidate action:
        {json.dumps(candidate, sort_keys=True)}

        Observation summary:
        {json.dumps(_compact_observation(obs), sort_keys=True)}

        Return exactly one JSON action object. Keep the same action type as the
        candidate unless it is invalid.
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        parsed = parse_action_json(response_text)
        if parsed.get("type") == candidate.get("type"):
            return sanitize_like_candidate(parsed, candidate)
    except Exception as exc:
        print(f"[DEBUG] model action selection failed: {exc}", flush=True)
    return dict(candidate)


def parse_action_json(response_text: str) -> dict:
    match = ACTION_JSON_RE.search(response_text or "")
    if not match:
        raise ValueError("model response did not contain a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict) or "type" not in parsed:
        raise ValueError("model response was not an action object")
    return parsed


def sanitize_like_candidate(parsed: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    action = dict(candidate)
    if candidate.get("type") == "publish_sitrep":
        payload = parsed.get("payload")
        if isinstance(payload, Mapping):
            candidate_payload = dict(candidate.get("payload", {}))
            summary = payload.get("summary_text")
            if isinstance(summary, str) and summary.strip():
                candidate_payload["summary_text"] = summary.strip()[:800]
            action["payload"] = candidate_payload
    return action


def grade_from_observation(obs: Mapping[str, Any], rewards: Iterable[float]) -> float:
    metadata = obs.get("metadata") or {}
    if "terminal_score" in metadata:
        return float(metadata["terminal_score"])
    if "score" in metadata:
        return float(metadata["score"])
    reward_total = sum(float(reward) for reward in rewards)
    return max(0.01, min(0.99, 0.35 + reward_total))


def run_task(client: Any, task_id: str) -> Tuple[float, List[float]]:
    log_start(task=task_id, client=client)
    memory = EpisodeMemory()
    rewards: List[float] = []
    step_count = 0
    final_obs: Dict[str, Any] = {}

    try:
        reset_response = env_reset(task_id)
        obs = reset_response.get("observation") or reset_response
        done = bool(reset_response.get("done", False) or obs.get("done", False))
        final_obs = dict(obs)
        memory.update_from_observation(obs)

        max_steps = int(os.getenv("MAX_STEPS", str(TASK_CONFIGS[task_id]["episode_cap"])))
        for step in range(1, max_steps + 1):
            if done:
                break
            candidate = make_policy_action(task_id, obs, memory)
            action = choose_action(client, task_id, obs, candidate)
            response = env_step(action)
            next_obs = response.get("observation") or response
            reward = float(response.get("reward") or next_obs.get("reward") or 0.0)
            done = bool(response.get("done", False) or next_obs.get("done", False))

            memory.remember_action(action)
            memory.update_from_observation(next_obs)
            rewards.append(reward)
            step_count = step
            final_obs = dict(next_obs)

            error = _error_from_observation(next_obs)
            log_step(
                step=step,
                action=str(action.get("type", "unknown")),
                reward=reward,
                done=done,
                error=error,
            )
            obs = next_obs

    except Exception as exc:
        print(f"[DEBUG] episode error for task={task_id}: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]

    score = grade_from_observation(final_obs, rewards) if final_obs else 0.01
    score = max(0.01, min(0.99, score))
    log_end(success=score >= SUCCESS_THRESHOLD, steps=step_count, score=score, rewards=rewards)
    return score, rewards


def selected_tasks() -> List[str]:
    requested = (
        os.getenv("TASK_ID")
        or os.getenv("TASK_NAME")
        or os.getenv("OPENENV_TASK_ID")
        or ""
    ).strip()
    if not requested:
        return list(TASK_TIERS)
    if requested in TASK_TIERS:
        return [requested]
    if requested in TASK_BY_TIER:
        return [TASK_BY_TIER[requested]]
    raise ValueError(
        f"Unknown task '{requested}'. Expected one of: "
        f"{', '.join(list(TASK_TIERS) + list(TASK_BY_TIER))}"
    )


def build_openai_client() -> Any:
    if not USE_LLM:
        return None
    if OpenAI is None:
        print("[DEBUG] openai package unavailable; using deterministic policy", flush=True)
        return None
    if not API_KEY or not MODEL_NAME:
        print("[DEBUG] API key/model missing; using deterministic policy", flush=True)
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def main() -> None:
    client = build_openai_client()
    policy = "llm" if client is not None else "deterministic"
    active_model = MODEL_NAME if client is not None else "none"
    print(
        f"[DEBUG] env={ENV_URL} policy={policy} model={active_model} "
        f"api_base={API_BASE_URL} hf_token_set={str(bool(API_KEY)).lower()} "
        f"use_llm={str(bool(client)).lower()}",
        flush=True,
    )
    results: Dict[str, Dict[str, Any]] = {}
    for task_id in selected_tasks():
        score, rewards = run_task(client, task_id)
        results[task_id] = {"score": score, "rewards": rewards}

    print("\n" + "=" * 72, flush=True)
    print(f"{'task':<32} {'tier':<8} {'score':>8}  rewards", flush=True)
    print("-" * 72, flush=True)
    for task_id, data in results.items():
        rewards_str = ", ".join(f"{reward:.2f}" for reward in data["rewards"])
        print(
            f"{task_id:<32} {TASK_TIERS[task_id]:<8} {data['score']:>8.3f}  "
            f"[{rewards_str}]",
            flush=True,
        )
    print("=" * 72, flush=True)


def _true_zone_ids(memory: EpisodeMemory) -> set[str]:
    return {
        str(report.get("zone_id", ""))
        for report_id, report in memory.known_reports.items()
        if report_id in memory.true_report_ids and report.get("zone_id")
    }


def _supporting_true_reports(memory: EpisodeMemory, zone_id: str) -> List[str]:
    report_ids = [
        report_id
        for report_id, report in memory.known_reports.items()
        if report_id in memory.true_report_ids and report.get("zone_id") == zone_id
    ]
    return sorted(report_ids)


def _required_unit_types(zone: Mapping[str, Any]) -> set[str]:
    required = zone.get("required_unit_types") or []
    if required:
        return {str(unit_type) for unit_type in required}
    return set(INCIDENT_UNIT_TYPES.get(str(zone.get("incident_type", "")), {"recon_drone"}))


def _best_available_unit(
    memory: EpisodeMemory,
    unit_type: str,
    zone: Mapping[str, Any],
) -> Optional[dict]:
    candidates = []
    zone_district = zone.get("district_id")
    for unit in memory.known_units.values():
        if unit.get("unit_id") in memory.allocated_unit_ids:
            continue
        if unit.get("unit_type") != unit_type:
            continue
        if unit.get("status") != "available":
            continue
        candidates.append(unit)
    if not candidates:
        return None
    candidates.sort(
        key=lambda unit: (
            0 if unit.get("district_id") in {None, zone_district} else 1,
            int(unit.get("fatigue", 0) or 0),
            int(unit.get("travel_cost", 1) or 1),
            str(unit.get("unit_id", "")),
        )
    )
    return dict(candidates[0])


def _unit_type_matches_zone(unit_type: str, zone: Mapping[str, Any]) -> bool:
    return unit_type in _required_unit_types(zone)


def _zone_sort_key(zone: Mapping[str, Any]) -> Tuple[int, int, str]:
    return (
        int(zone.get("deadline_steps", 99) or 99),
        -int(zone.get("severity", 0) or 0) * int(zone.get("population_at_risk", 0) or 0),
        str(zone.get("zone_id", "")),
    )


def _priority_for_zone(zone: Mapping[str, Any]) -> str:
    severity = int(zone.get("severity", 0) or 0)
    if severity >= 5:
        return "critical"
    if severity >= 4:
        return "high"
    if severity <= 2:
        return "low"
    return "normal"


def _verification_method(report: Mapping[str, Any]) -> str:
    source = report.get("source")
    confidence = report.get("confidence")
    if confidence == "sensor_confirmed":
        return "sensor_review"
    if source == "official":
        return "official_confirmation"
    if source in {"field_team", "media"}:
        return "cross_check"
    return "contact_source"


def _compact_observation(obs: Mapping[str, Any]) -> dict:
    return {
        "time_step": obs.get("time_step"),
        "metadata": obs.get("metadata", {}),
        "visible_zones": [
            {
                "zone_id": zone.get("zone_id"),
                "incident_type": zone.get("incident_type"),
                "severity": zone.get("severity"),
                "deadline_steps": zone.get("deadline_steps"),
                "access_status": zone.get("access_status"),
                "district_id": zone.get("district_id"),
                "required_unit_types": zone.get("required_unit_types"),
            }
            for zone in (obs.get("visible_zones") or [])
        ],
        "reports": [
            {
                "report_id": report.get("report_id"),
                "zone_id": report.get("zone_id"),
                "report_type": report.get("report_type"),
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
                "fatigue": unit.get("fatigue"),
                "district_id": unit.get("district_id"),
                "mutual_aid_unlock_step": unit.get("mutual_aid_unlock_step"),
            }
            for unit in (obs.get("resources") or [])
        ],
        "incident_log": obs.get("incident_log", []),
    }


def _error_from_observation(obs: Mapping[str, Any]) -> Optional[str]:
    logs = [str(item) for item in (obs.get("incident_log") or [])]
    negative = [
        log
        for log in logs
        if any(term in log.lower() for term in ("rejected", "invalid", "error"))
    ]
    return "; ".join(negative) if negative else None


if __name__ == "__main__":
    main()
