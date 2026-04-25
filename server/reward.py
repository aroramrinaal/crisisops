"""Step-level Crisisops training rewards."""

from collections.abc import Mapping
from typing import Any

try:
    from ..models import CrisisopsAction
    from .rules import INCIDENT_UNIT_TYPES
except ImportError:
    from models import CrisisopsAction
    from server.rules import INCIDENT_UNIT_TYPES


def compute_step_reward(
    action: CrisisopsAction | Mapping[str, Any],
    prev_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    hidden_state: Mapping[str, Any],
) -> float:
    """Return plan-aligned reward for one transition."""

    payload = _action_payload(action)
    action_type = payload.get("type")
    reward = -0.01

    if action_type == "verify_report":
        report_id = payload.get("report_id")
        if _report_truth(report_id, hidden_state) is True:
            reward += 0.05

    elif action_type == "flag_false_alarm":
        report_id = payload.get("report_id")
        reward += 0.05 if _report_truth(report_id, hidden_state) is False else -0.10

    elif action_type in {"allocate_unit", "issue_evacuation"}:
        if _acts_on_unverified_non_sensor_report(payload, hidden_state):
            reward -= 0.10
        if action_type == "allocate_unit" and _wrong_unit_type(payload, hidden_state):
            reward -= 0.15

    elif action_type == "reroute_unit":
        route = payload.get("route", {})
        if route.get("status") in {"blocked", "unsafe"}:
            reward -= 0.20

    elif action_type == "noop":
        if new_state.get("consecutive_noop_count", 0) > 2:
            reward -= 0.02

    if _targets_blocked_zone(payload, hidden_state):
        reward -= 0.20
    if _shelter_overfilled(payload, hidden_state):
        reward -= 0.10

    reward += 0.30 * len(
        _new_resolved_before_deadline(prev_state, new_state, hidden_state)
    )
    reward -= 0.50 * len(_new_deadline_misses(prev_state, new_state))
    return float(reward)


def _wrong_unit_type(
    payload: Mapping[str, Any], hidden_state: Mapping[str, Any]
) -> bool:
    units_by_id = hidden_state.get("units_by_id", {})
    zones_by_id = hidden_state.get("zones_by_id", {})
    unit = units_by_id.get(payload.get("unit_id"), {})
    zone = zones_by_id.get(payload.get("zone_id"), {})
    if not unit or not zone:
        return False
    return unit.get("unit_type") not in INCIDENT_UNIT_TYPES.get(
        zone.get("incident_type"), set()
    )


def _report_truth(report_id: str | None, hidden_state: Mapping[str, Any]) -> bool | None:
    if report_id is None:
        return None
    if report_id in hidden_state.get("report_truth", {}):
        return hidden_state["report_truth"][report_id]
    return hidden_state.get("event_report_truth", {}).get(report_id)


def _acts_on_unverified_non_sensor_report(
    payload: Mapping[str, Any], hidden_state: Mapping[str, Any]
) -> bool:
    verified = set(hidden_state.get("verified_report_ids", []))
    reports_by_id = hidden_state.get("reports_by_id", {})
    report_ids = set(payload.get("report_ids", []))
    if not report_ids and payload.get("zone_id"):
        report_ids = {
            report_id
            for report_id, report in reports_by_id.items()
            if report.get("zone_id") == payload.get("zone_id")
        }
    return any(
        reports_by_id.get(report_id, {}).get("confidence") != "sensor_confirmed"
        and report_id not in verified
        for report_id in report_ids
    )


def _targets_blocked_zone(
    payload: Mapping[str, Any], hidden_state: Mapping[str, Any]
) -> bool:
    if payload.get("type") not in {
        "allocate_unit",
        "issue_evacuation",
        "dispatch_supplies",
    }:
        return False
    zone_id = payload.get("zone_id") or payload.get("destination_zone_id")
    zone = hidden_state.get("zones_by_id", {}).get(zone_id, {})
    return zone.get("access_status") == "blocked"


def _shelter_overfilled(
    payload: Mapping[str, Any], hidden_state: Mapping[str, Any]
) -> bool:
    if payload.get("type") != "issue_evacuation":
        return False
    zone_id = payload.get("zone_id")
    shelter_id = payload.get("destination_shelter_id")
    if not zone_id or not shelter_id:
        return False
    zone = hidden_state.get("zones_by_id", {}).get(zone_id, {})
    shelter = hidden_state.get("shelters_by_id", {}).get(shelter_id, {})
    return bool(
        zone
        and shelter
        and zone.get("population_at_risk", 0)
        > shelter.get("capacity_available", 0)
    )


def _new_resolved_before_deadline(
    prev_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    hidden_state: Mapping[str, Any],
) -> set[str]:
    prev_steps = set(prev_state.get("first_correct_allocation_steps", {}))
    new_steps = dict(new_state.get("first_correct_allocation_steps", {}))
    deadlines = hidden_state.get("zone_deadlines", {})
    return {
        zone_id
        for zone_id in set(new_steps) - prev_steps
        if new_steps[zone_id] < deadlines.get(zone_id, -1)
        and zone_id in set(hidden_state.get("critical_zone_ids", []))
    }


def _new_deadline_misses(
    prev_state: Mapping[str, Any], new_state: Mapping[str, Any]
) -> set[str]:
    return set(new_state.get("deadline_missed_zone_ids", [])) - set(
        prev_state.get("deadline_missed_zone_ids", [])
    )


def _action_payload(action: CrisisopsAction | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, CrisisopsAction):
        return action.root.model_dump(mode="json")
    if "root" in action and isinstance(action["root"], Mapping):
        return dict(action["root"])
    return dict(action)
