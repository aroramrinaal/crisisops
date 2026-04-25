"""Step-level Crisisops training rewards."""

from collections.abc import Mapping
from typing import Any

try:
    from ..models import CrisisopsAction
    from .rules import is_unsafe_action, unit_type_for_incident
except ImportError:
    from models import CrisisopsAction
    from server.rules import is_unsafe_action, unit_type_for_incident


def compute_step_reward(
    action: CrisisopsAction | Mapping[str, Any],
    prev_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    hidden_state: Mapping[str, Any],
) -> float:
    """Return intentionally unbounded reward for one transition."""

    payload = _action_payload(action)
    action_type = payload.get("type")
    reward = -0.02

    if is_unsafe_action(payload, hidden_state):
        reward -= 1.25

    if action_type == "verify_report":
        report_id = payload.get("report_id")
        if hidden_state.get("report_truth", {}).get(report_id) is True:
            reward += 0.45
        else:
            reward += 0.25

    elif action_type == "flag_false_alarm":
        report_id = payload.get("report_id")
        reward += 0.9 if hidden_state.get("report_truth", {}).get(report_id) is False else -1.1

    elif action_type == "request_recon":
        zone_id = payload.get("zone_id")
        reward += 0.35 if zone_id in hidden_state.get("incident_zones", []) else 0.05

    elif action_type == "allocate_unit":
        reward += _allocation_reward(payload, hidden_state)

    elif action_type == "reroute_unit":
        route = payload.get("route", {})
        reward += 0.55 if route.get("status") in {"open", "congested"} else -0.8

    elif action_type == "issue_evacuation":
        reward += 0.75 if payload.get("zone_id") in hidden_state.get("incident_zones", []) else -0.9

    elif action_type == "open_shelter":
        shelter = payload.get("shelter", {})
        reward += 0.5 if shelter.get("capacity_available", 0) > 0 else -0.5

    elif action_type == "dispatch_supplies":
        reward += 0.65 if payload.get("destination_zone_id") in hidden_state.get("incident_zones", []) else -0.35

    elif action_type == "publish_sitrep":
        payload_data = payload.get("payload", {})
        reward += 0.15 * len(payload_data.get("verified_report_ids", []))
        reward -= 0.1 * len(payload_data.get("pending_verification_report_ids", []))

    elif action_type == "noop":
        pending_reports = _pending_reports(new_state, hidden_state)
        reward += 0.2 if pending_reports else -0.15

    reward += _progress_delta(prev_state, new_state)
    return float(reward)


def _allocation_reward(payload: Mapping[str, Any], hidden_state: Mapping[str, Any]) -> float:
    report_ids = payload.get("report_ids", [])
    reports_by_id = hidden_state.get("reports_by_id", {})
    for report_id in report_ids:
        report = reports_by_id.get(report_id)
        if not report:
            continue
        expected = unit_type_for_incident(report.get("report_type", "other"))
        if hidden_state.get("report_truth", {}).get(report_id) and payload.get("task"):
            return 1.0 if expected in {payload.get("task"), payload.get("unit_type")} else 0.55
    return 0.25


def _pending_reports(
    state: Mapping[str, Any], hidden_state: Mapping[str, Any]
) -> list[str]:
    verified = set(state.get("verified_report_ids", hidden_state.get("verified_report_ids", [])))
    report_truth = hidden_state.get("report_truth", {})
    return [report_id for report_id in report_truth if report_id not in verified]


def _progress_delta(prev_state: Mapping[str, Any], new_state: Mapping[str, Any]) -> float:
    prev_allocations = len(prev_state.get("allocated_unit_ids", []))
    new_allocations = len(new_state.get("allocated_unit_ids", []))
    prev_verified = len(prev_state.get("verified_report_ids", []))
    new_verified = len(new_state.get("verified_report_ids", []))
    return 0.1 * (new_allocations - prev_allocations) + 0.05 * (
        new_verified - prev_verified
    )


def _action_payload(action: CrisisopsAction | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, CrisisopsAction):
        return action.root.model_dump(mode="json")
    if "root" in action and isinstance(action["root"], Mapping):
        return dict(action["root"])
    return dict(action)
