"""
Baseline inference script for the CrisisOps OpenEnv environment.

The script lets an OpenAI-compatible model drive the environment when
credentials are available, and uses a deterministic incident-command policy only
as a fallback when the model call or returned JSON action fails.

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
    SUCCESS_THRESHOLD  Score needed for success=true. Defaults to 0.75.
    MAX_STEPS      Per-episode safety cap. Defaults to the task episode cap.

STDOUT format:
    [START] task=<task_id> env=crisisops policy=<deterministic|llm> model=<model_name|none>
    [STEP]  step=<n> action=<action_type> source=<source> reward=<0.00> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<0.000> sources=<counts> rewards=<r1,...,rn>
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
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.75"))

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
    JSON object for the next environment action.

    DEFAULT BEHAVIOR: copy the `Recommended action` block VERBATIM. The
    deterministic policy that produced it already accounts for verification
    status, unit-type matching, blocked zones, deadlines, and sitrep timing.
    Deviate ONLY if the observation reveals something the recommendation
    clearly missed (e.g. it suggests allocating to a zone that just turned
    blocked). Do NOT replace allocate_unit with verify_report just because a
    rule mentions verification — the recommender already verified what
    needed verifying.

    Hard rules (these directly affect reward, listed for tie-breaking only):
    - allocate_unit / issue_evacuation against an unverified non-sensor report
      costs -0.10. The recommender will not do this; do not introduce it.
    - allocate_unit whose unit_type is not in the zone's required_unit_types
      costs -0.15.
    - Targeting a zone with access_status == "blocked" costs -0.20.
    - Resolving a critical zone with the right unit BEFORE its deadline_steps
      pays +0.30. Missing a deadline costs -0.50, so do NOT stall on
      verification once at least one true zone has a matching available unit.
    - Flag false_alarm only after a verification disputed the report.
    - publish_sitrep ends the episode — only do it when the recommender does.

    Use only IDs that appear in the visible observation. Return JSON only. No
    markdown fences. No prose outside the JSON.
    """
).strip()

TASK_BRIEFS: Dict[str, str] = {
    "single_zone_response": (
        "EASY tier — one district, one active incident zone, ~3 incoming reports, "
        "8-step episode cap. Verify the report stream, allocate the matching unit "
        "type to the true zone before its deadline, then publish a sitrep."
    ),
    "multi_zone_triage": (
        "MEDIUM tier — single district with multiple concurrent incident zones, "
        "~6 reports including some false alarms, 15-step cap. Triage by deadline "
        "and severity: verify before committing scarce units, flag disputed "
        "reports as false_alarm, then publish a sitrep when urgent work is done."
    ),
    "cascading_crisis": (
        "HARD tier — incidents cascade as the episode unfolds; ~10 reports stream "
        "in (some appear after step 12), 25-step cap. Earlier zones must be "
        "resolved before new ones arrive or you will start missing deadlines "
        "(-0.50 each). Verify non-sensor reports before allocation."
    ),
    "multi_district_coordination": (
        "EXPERT tier — multiple districts with limited mutual-aid units, ~16 "
        "reports streaming through step 20, 40-step cap. Comms in one district "
        "degrade mid-episode (sensor reports drop to lower confidence) so verify "
        "aggressively. Mutual-aid units unlock at mutual_aid_unlock_step. Match "
        "unit_type to required_unit_types and avoid blocked zones."
    ),
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
    {"type":"dispatch_supplies","supplies":{"water":100,"medical_kits":10},"destination_zone_id":"zone-1","priority":"high","unit_id":null,"destination_shelter_id":null}
    {"type":"flag_false_alarm","report_id":"report-1","rationale":"...","evidence":["..."]}
    {"type":"publish_sitrep","payload":{"incidents_confirmed":["report-1"],"incidents_resolved":["zone-1"],"unresolved_risks":[],"false_alarms_detected":[],"summary_text":"..."}}
    {"type":"noop","reason":"..."}

    Allowed values:
    verification_method: cross_check, contact_source, field_recon, sensor_review, official_confirmation
    priority/urgency: low, normal, high, critical
    task: rescue, medical, evacuation, fire_suppression, supply_delivery, recon, route_clearance
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


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
    source: str,
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} source={source} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
    action_sources: List[str],
) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    source_counts = _format_source_counts(action_sources)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} sources={source_counts} rewards={rewards_str}",
        flush=True,
    )


def _format_source_counts(action_sources: Iterable[str]) -> str:
    counts: Dict[str, int] = {}
    for source in action_sources:
        counts[source] = counts.get(source, 0) + 1
    if not counts:
        return "none"
    return ",".join(f"{source}:{counts[source]}" for source in sorted(counts))


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


def env_step(session_id: str, action: Mapping[str, Any]) -> dict:
    return post_json("/step", {"session_id": session_id, "action": dict(action)})


def env_state(session_id: str) -> dict:
    request = Request(f"{ENV_URL}/state?session_id={session_id}", method="GET")
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from /state: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {ENV_URL}/state: {exc}") from exc


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
    fallback_action: Mapping[str, Any],
    history: List[Dict[str, Any]],
) -> Tuple[dict, str]:
    if client is None or not USE_LLM:
        return dict(fallback_action), "deterministic"

    brief = TASK_BRIEFS.get(task_id, "(no brief)")
    history_lines = _format_history_lines(history)

    prompt = textwrap.dedent(
        f"""
        Task: {task_id}
        Brief: {brief}
        Time step: {obs.get("time_step")}
        Episode cap: {(obs.get("metadata") or {}).get("episode_cap", TASK_CONFIGS[task_id]["episode_cap"])}

        Action contract:
        {ACTION_FORMAT_PROMPT}

        Recent steps (most recent last):
        {history_lines}

        Current observation:
        {json.dumps(_compact_observation(obs), sort_keys=True)}

        Recommended action (deterministic policy's pick — copy verbatim unless
        the observation gives a clear reason to deviate):
        {json.dumps(dict(fallback_action), sort_keys=True)}

        Return exactly one JSON object.
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
        return sanitize_model_action(parsed), "llm"
    except Exception as exc:
        print(f"[DEBUG] model action selection failed; using fallback: {exc}", flush=True)
    return dict(fallback_action), "fallback"


def parse_action_json(response_text: str) -> dict:
    match = ACTION_JSON_RE.search(response_text or "")
    if not match:
        raise ValueError("model response did not contain a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict) or "type" not in parsed:
        raise ValueError("model response was not an action object")
    return parsed


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
                "false_alarms_detected": _string_list(
                    payload.get("false_alarms_detected")
                ),
                "summary_text": _text_value(
                    payload.get("summary_text"),
                    "Situation report published.",
                )[:800],
            },
        }

    if action_type == "noop":
        return {
            "type": "noop",
            "reason": _text_value(parsed.get("reason"), "Waiting for more evidence."),
        }

    raise ValueError(f"unknown action type: {action_type}")


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


def grade_from_observation(obs: Mapping[str, Any], rewards: Iterable[float]) -> float:
    metadata = obs.get("metadata") or {}
    if "terminal_score" in metadata:
        return float(metadata["terminal_score"])
    if "score" in metadata:
        return float(metadata["score"])
    reward_total = sum(float(reward) for reward in rewards)
    return max(0.01, min(0.99, 0.35 + reward_total))


def _format_history_lines(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "  (none yet)"
    lines = []
    for entry in history[-5:]:
        error = entry.get("error") or "ok"
        lines.append(
            f"  step={entry['step']} action={entry['action']} "
            f"reward={entry['reward']:+.2f} done={str(entry['done']).lower()} "
            f"error={error}"
        )
    return "\n".join(lines)


def run_task(client: Any, task_id: str) -> Tuple[float, List[float], List[str]]:
    log_start(task=task_id, client=client)
    memory = EpisodeMemory()
    rewards: List[float] = []
    action_sources: List[str] = []
    history: List[Dict[str, Any]] = []
    step_count = 0
    final_obs: Dict[str, Any] = {}

    try:
        reset_response = env_reset(task_id)
        session_id = str(reset_response.get("session_id", ""))
        if not session_id:
            raise RuntimeError("Reset response did not include session_id")
        obs = reset_response.get("observation") or reset_response
        done = bool(reset_response.get("done", False) or obs.get("done", False))
        final_obs = dict(obs)
        memory.update_from_observation(obs)
        env_state(session_id)

        max_steps = int(
            os.getenv("MAX_STEPS", str(TASK_CONFIGS[task_id]["episode_cap"]))
        )
        for step in range(1, max_steps + 1):
            if done:
                break
            fallback_action = make_policy_action(task_id, obs, memory)
            action, action_source = choose_action(
                client, task_id, obs, fallback_action, history
            )
            response = env_step(session_id, action)
            next_obs = response.get("observation") or response
            reward = float(response.get("reward") or next_obs.get("reward") or 0.0)
            done = bool(response.get("done", False) or next_obs.get("done", False))

            memory.remember_action(action)
            memory.update_from_observation(next_obs)
            rewards.append(reward)
            action_sources.append(action_source)
            step_count = step
            final_obs = dict(next_obs)

            error = _error_from_observation(next_obs)
            history.append(
                {
                    "step": step,
                    "action": str(action.get("type", "unknown")),
                    "reward": reward,
                    "done": done,
                    "error": error,
                }
            )
            log_step(
                step=step,
                action=str(action.get("type", "unknown")),
                reward=reward,
                done=done,
                error=error,
                source=action_source,
            )
            obs = next_obs

    except Exception as exc:
        print(f"[DEBUG] episode error for task={task_id}: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]

    score = grade_from_observation(final_obs, rewards) if final_obs else 0.01
    score = max(0.01, min(0.99, score))
    log_end(
        success=score >= SUCCESS_THRESHOLD,
        steps=step_count,
        score=score,
        rewards=rewards,
        action_sources=action_sources,
    )
    return score, rewards, action_sources


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
        f"use_llm={str(bool(client)).lower()} transport=http-session",
        flush=True,
    )
    results: Dict[str, Dict[str, Any]] = {}
    for task_id in selected_tasks():
        score, rewards, action_sources = run_task(client, task_id)
        results[task_id] = {
            "score": score,
            "rewards": rewards,
            "sources": action_sources,
        }

    print("\n" + "=" * 72, flush=True)
    print(f"{'task':<32} {'tier':<8} {'score':>8}  {'sources':<20} rewards", flush=True)
    print("-" * 72, flush=True)
    for task_id, data in results.items():
        rewards_str = ", ".join(f"{reward:.2f}" for reward in data["rewards"])
        source_counts = _format_source_counts(data["sources"])
        print(
            f"{task_id:<32} {TASK_TIERS[task_id]:<8} {data['score']:>8.3f}  "
            f"{source_counts:<20} [{rewards_str}]",
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
                "name": zone.get("name"),
                "incident_type": zone.get("incident_type"),
                "severity": zone.get("severity"),
                "population_at_risk": zone.get("population_at_risk"),
                "deadline_steps": zone.get("deadline_steps"),
                "access_status": zone.get("access_status"),
                "district_id": zone.get("district_id"),
                "required_unit_types": sorted(zone.get("required_unit_types") or []),
                "shelter_id": (zone.get("shelter") or {}).get("shelter_id"),
                "shelter_capacity_available": (zone.get("shelter") or {}).get(
                    "capacity_available"
                ),
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
