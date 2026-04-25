"""Rule-derived Crisisops domain logic.

This module is the single source of truth for scenario generation, rewards, and
grading. Keep hand-authored answers out of those consumers.
"""

from collections.abc import Mapping, Sequence
from typing import Any

try:
    from ..models import (
        CrisisopsAction,
        Report,
        SitrepPayload,
        Zone,
    )
except ImportError:
    from models import (
        CrisisopsAction,
        Report,
        SitrepPayload,
        Zone,
    )


INCIDENT_UNIT_TYPES = {
    "flood": "rescue",
    "fire": "fire",
    "collapse": "rescue",
    "medical": "medical",
    "infrastructure": "police",
    "shelter": "supply",
    "resource": "supply",
    "other": "recon",
}

RISK_ORDER = {
    "unknown": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}


def unit_type_for_incident(incident_type: str) -> str:
    """Return the preferred response unit type for an incident category."""

    return INCIDENT_UNIT_TYPES.get(incident_type, "recon")


def priority_key(zone: Zone) -> tuple[int, int, str]:
    """Sort key for highest-risk, highest-population zones first."""

    return (
        -RISK_ORDER.get(zone.risk_level, 0),
        -(zone.population_estimate or 0),
        zone.zone_id,
    )


def is_unsafe_action(
    action: CrisisopsAction | Mapping[str, Any],
    hidden_state: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether an action violates pause-verify-allocate discipline."""

    payload = _action_payload(action)
    action_type = payload.get("type")
    hidden_state = hidden_state or {}
    report_truth = hidden_state.get("report_truth", {})
    verified_report_ids = set(hidden_state.get("verified_report_ids", set()))
    incident_zones = set(hidden_state.get("incident_zones", set()))

    if action_type in {"verify_report", "request_recon", "noop", "publish_sitrep"}:
        return False

    if action_type == "flag_false_alarm":
        report_id = payload.get("report_id")
        return bool(report_id and report_truth.get(report_id) is True)

    report_ids = set(payload.get("report_ids", []))
    if report_ids:
        if any(report_truth.get(report_id) is False for report_id in report_ids):
            return True
        if verified_report_ids and not report_ids.issubset(verified_report_ids):
            return True

    zone_id = payload.get("zone_id") or payload.get("destination_zone_id")
    if action_type in {"allocate_unit", "issue_evacuation"}:
        return bool(zone_id and incident_zones and zone_id not in incident_zones)

    if action_type == "dispatch_supplies":
        supplies = payload.get("supplies", {})
        return not supplies or any(amount <= 0 for amount in supplies.values())

    if action_type == "open_shelter":
        shelter = payload.get("shelter", {})
        return shelter.get("capacity_available", 0) <= 0

    if action_type == "reroute_unit":
        route = payload.get("route", {})
        return route.get("status") in {"blocked", "unsafe"}

    return False


def compute_optimal_plan(scenario: Mapping[str, Any]) -> list[CrisisopsAction]:
    """Compute the deterministic reference plan for a generated scenario."""

    zones: Sequence[Zone] = scenario["zones"]
    units = list(scenario["units"])
    reports: Sequence[Report] = scenario["reports"]
    hidden_truth: Mapping[str, Any] = scenario["hidden_truth"]
    report_truth: Mapping[str, bool] = hidden_truth["report_truth"]

    actions: list[CrisisopsAction] = []
    sorted_zones = sorted(zones, key=priority_key)
    zone_rank = {zone.zone_id: index for index, zone in enumerate(sorted_zones)}
    sorted_reports = sorted(
        reports,
        key=lambda report: (
            zone_rank.get(report.zone_id, len(zone_rank)),
            -RISK_ORDER.get(report.severity, 0),
            report.report_id,
        ),
    )

    for report in sorted_reports:
        actions.append(
            _action(
                {
                    "type": "verify_report",
                    "report_id": report.report_id,
                    "verification_method": "cross_check",
                    "rationale": "Confirm noisy report before allocating scarce units.",
                }
            )
        )
        if report_truth[report.report_id] is False:
            actions.append(
                _action(
                    {
                        "type": "flag_false_alarm",
                        "report_id": report.report_id,
                        "rationale": "Ground truth checks contradict the report.",
                        "evidence": ["seeded_truth_table"],
                    }
                )
            )

    assigned_units: set[str] = set()
    true_reports = [report for report in sorted_reports if report_truth[report.report_id]]
    for zone in sorted_zones:
        zone_reports = [report for report in true_reports if report.zone_id == zone.zone_id]
        if not zone_reports:
            continue

        if zone.risk_level in {"high", "critical"} and zone.shelter is not None:
            shelter = zone.shelter.model_copy(update={"status": "open"})
            actions.append(
                _action(
                    {
                        "type": "open_shelter",
                        "shelter": shelter.model_dump(),
                        "reason": f"Prepare shelter capacity for {zone.risk_level} risk zone.",
                    }
                )
            )

        for report in zone_reports:
            unit_type = unit_type_for_incident(report.report_type)
            unit = next(
                (
                    unit
                    for unit in units
                    if unit.unit_id not in assigned_units
                    and (unit.unit_type == unit_type or unit_type in unit.capabilities)
                    and unit.status == "available"
                ),
                None,
            )
            if unit is None:
                continue
            assigned_units.add(unit.unit_id)
            actions.append(
                _action(
                    {
                        "type": "allocate_unit",
                        "unit_id": unit.unit_id,
                        "zone_id": zone.zone_id,
                        "task": _task_for_report(report.report_type),
                        "priority": _priority_for_zone(zone),
                        "report_ids": [report.report_id],
                    }
                )
            )

        if zone.risk_level == "critical":
            actions.append(
                _action(
                    {
                        "type": "dispatch_supplies",
                        "supplies": {"water": 100, "medical_kits": 12},
                        "destination_zone_id": zone.zone_id,
                        "priority": "critical",
                        "destination_shelter_id": (
                            zone.shelter.shelter_id if zone.shelter else None
                        ),
                    }
                )
            )

    actions.append(
        _action(
            {
                "type": "publish_sitrep",
                "payload": SitrepPayload(
                    summary="Prioritize verified high-risk zones and conserve resources.",
                    priorities=[zone.zone_id for zone in sorted_zones[:3]],
                    verified_report_ids=[
                        report.report_id for report in true_reports
                    ],
                    pending_verification_report_ids=[],
                    allocations=[
                        _action_payload(action).get("unit_id")
                        for action in actions
                        if _action_payload(action).get("type") == "allocate_unit"
                    ],
                    next_actions=["monitor_routes", "refresh_report_truth"],
                ).model_dump(),
            }
        )
    )

    return actions


def _task_for_report(report_type: str) -> str:
    return {
        "flood": "rescue",
        "fire": "fire_suppression",
        "collapse": "rescue",
        "medical": "medical",
        "infrastructure": "route_clearance",
        "shelter": "supply_delivery",
        "resource": "supply_delivery",
    }.get(report_type, "recon")


def _priority_for_zone(zone: Zone) -> str:
    if zone.risk_level == "critical":
        return "critical"
    if zone.risk_level == "high":
        return "high"
    if zone.risk_level == "low":
        return "low"
    return "normal"


def _action(data: Mapping[str, Any]) -> CrisisopsAction:
    return CrisisopsAction.model_validate(dict(data))


def _action_payload(action: CrisisopsAction | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, CrisisopsAction):
        return action.root.model_dump(mode="json")
    if "root" in action and isinstance(action["root"], Mapping):
        return dict(action["root"])
    return dict(action)
