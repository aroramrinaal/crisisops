"""Rule-derived Crisisops domain logic."""

from collections.abc import Mapping, Sequence
from typing import Any

try:
    from ..models import CrisisopsAction, Report, SitrepPayload, Unit, Zone
except ImportError:
    from models import CrisisopsAction, Report, SitrepPayload, Unit, Zone


INCIDENT_UNIT_TYPES: dict[str, set[str]] = {
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


def unit_types_for_incident(incident_type: str) -> set[str]:
    """Return plan-required unit types for an incident category."""

    return set(INCIDENT_UNIT_TYPES.get(incident_type, {"recon_drone"}))


def unit_type_for_incident(incident_type: str) -> str:
    """Return one preferred unit type for legacy callers."""

    return sorted(unit_types_for_incident(incident_type))[0]


def priority_key(zone: Zone) -> tuple[int, int, str]:
    """Plan section 3.3.2 priority key: deadline, severity-population, id."""

    return (
        zone.deadline_steps,
        -(zone.severity * zone.population_at_risk),
        zone.zone_id,
    )


def is_unsafe_action(
    action: CrisisopsAction | Mapping[str, Any],
    hidden_state: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether an action violates the plan's safety preconditions."""

    payload = _action_payload(action)
    action_type = payload.get("type")
    hidden_state = hidden_state or {}

    if action_type in {"verify_report", "request_recon", "noop", "publish_sitrep"}:
        return False

    if action_type == "flag_false_alarm":
        report_id = payload.get("report_id")
        return bool(report_id and _report_truth(report_id, hidden_state) is True)

    if _targets_blocked_zone(payload, hidden_state):
        return True

    if action_type == "reroute_unit":
        route = payload.get("route", {})
        return route.get("status") in {"blocked", "unsafe"}

    if action_type in {"allocate_unit", "issue_evacuation"}:
        if _acts_on_unverified_non_sensor_report(payload, hidden_state):
            return True
        zone_id = payload.get("zone_id")
        incident_zones = set(hidden_state.get("incident_zones", []))
        if zone_id and incident_zones and zone_id not in incident_zones:
            return True

    if action_type == "dispatch_supplies":
        supplies = payload.get("supplies", {})
        return not supplies or any(amount <= 0 for amount in supplies.values())

    if action_type == "issue_evacuation":
        return _overfills_shelter(payload, hidden_state)

    if action_type == "open_shelter":
        shelter = payload.get("shelter", {})
        return shelter.get("capacity_available", 0) <= 0

    return False


def compute_optimal_plan(scenario: Mapping[str, Any]) -> list[CrisisopsAction]:
    """Compute the deterministic reference plan for a generated scenario."""

    zones: Sequence[Zone] = scenario["zones"]
    units: list[Unit] = list(scenario["units"])
    hidden_truth: Mapping[str, Any] = scenario["hidden_truth"]
    report_truth: Mapping[str, bool] = hidden_truth["report_truth"]
    reports = [
        *scenario["reports"],
        *_event_reports_from_hidden(hidden_truth),
    ]

    actions: list[CrisisopsAction] = []
    sorted_zones = sorted(zones, key=priority_key)
    zone_rank = {zone.zone_id: index for index, zone in enumerate(sorted_zones)}
    sorted_reports = sorted(
        reports,
        key=lambda report: (
            zone_rank.get(report.zone_id, len(zone_rank)),
            report.reveal_at_step,
            report.report_id,
        ),
    )
    reports_by_zone = {
        zone.zone_id: [report for report in sorted_reports if report.zone_id == zone.zone_id]
        for zone in sorted_zones
    }
    true_reports_by_zone = {
        zone.zone_id: [
            report
            for report in reports_by_zone[zone.zone_id]
            if report_truth[report.report_id]
        ]
        for zone in sorted_zones
    }

    verified_report_ids = _sensor_confirmed_true_ids(sorted_reports, report_truth)
    flagged_false_ids: set[str] = set()
    assigned_units: set[str] = set()
    allocated_unit_types_by_zone: dict[str, set[str]] = {}
    resolved_zone_ids: set[str] = set()

    # Timeliness is measured by action index, so critical zones get a safe first
    # allocation before the slower full verification sweep.
    for zone in sorted_zones:
        if not _is_critical(zone) or not true_reports_by_zone[zone.zone_id]:
            continue
        unit = _next_unit_for_zone(units, assigned_units, zone)
        if unit is None:
            continue
        support_report = true_reports_by_zone[zone.zone_id][0]
        _append_verification_actions(
            actions,
            support_report,
            report_truth,
            verified_report_ids,
            flagged_false_ids,
        )
        _append_allocation_action(
            actions,
            unit,
            zone,
            [support_report.report_id],
            assigned_units,
            allocated_unit_types_by_zone,
            resolved_zone_ids,
        )

    for report in sorted_reports:
        _append_noops_until(actions, report.reveal_at_step)
        _append_verification_actions(
            actions,
            report,
            report_truth,
            verified_report_ids,
            flagged_false_ids,
        )

    for zone in sorted_zones:
        zone_reports = true_reports_by_zone[zone.zone_id]
        if not zone_reports:
            continue

        if _is_critical(zone):
            actions.append(
                _action(
                    {
                        "type": "issue_evacuation",
                        "zone_id": zone.zone_id,
                        "urgency": "critical",
                        "message": f"Evacuate {zone.name or zone.zone_id} before deadline.",
                        "route_id": zone.routes[0].route_id if zone.routes else None,
                        "destination_shelter_id": (
                            zone.shelter.shelter_id if zone.shelter else None
                        ),
                    }
                )
            )
            if zone.shelter is not None:
                shelter = zone.shelter.model_copy(update={"status": "open"})
                actions.append(
                    _action(
                        {
                            "type": "open_shelter",
                            "shelter": shelter.model_dump(mode="json"),
                            "reason": "Open shelter capacity for critical incident evacuation.",
                        }
                    )
                )

        allocated_types = allocated_unit_types_by_zone.setdefault(zone.zone_id, set())
        for unit_type in sorted(zone.required_unit_types - allocated_types):
            unit = _next_unit(units, assigned_units, unit_type)
            if unit is None:
                continue
            _append_allocation_action(
                actions,
                unit,
                zone,
                [report.report_id for report in zone_reports],
                assigned_units,
                allocated_unit_types_by_zone,
                resolved_zone_ids,
            )

        if _is_critical(zone):
            supply_unit = _next_unit(units, assigned_units, "supply_truck")
            actions.append(
                _action(
                    {
                        "type": "dispatch_supplies",
                        "supplies": {"water": 100, "medical_kits": 12},
                        "destination_zone_id": zone.zone_id,
                        "priority": "critical",
                        "unit_id": supply_unit.unit_id if supply_unit else None,
                        "destination_shelter_id": (
                            zone.shelter.shelter_id if zone.shelter else None
                        ),
                    }
                )
            )
            if supply_unit is not None:
                assigned_units.add(supply_unit.unit_id)

    true_reports = [report for report in sorted_reports if report_truth[report.report_id]]
    unresolved_zone_ids = sorted(set(hidden_truth.get("incident_zones", [])) - resolved_zone_ids)
    actions.append(
        _action(
            {
                "type": "publish_sitrep",
                "payload": SitrepPayload(
                    incidents_confirmed=[report.report_id for report in true_reports],
                    incidents_resolved=sorted(resolved_zone_ids),
                    unresolved_risks=unresolved_zone_ids,
                    false_alarms_detected=list(hidden_truth.get("false_report_ids", [])),
                    summary_text=(
                        "Verified incidents prioritized by deadline and population "
                        "at risk."
                    ),
                ).model_dump(mode="json"),
            }
        )
    )

    return actions


def _event_reports_from_hidden(hidden_truth: Mapping[str, Any]) -> list[Report]:
    return [
        Report.model_validate(report)
        for report in hidden_truth.get("event_reports_by_id", {}).values()
    ]


def _sensor_confirmed_true_ids(
    reports: Sequence[Report], report_truth: Mapping[str, bool]
) -> set[str]:
    return {
        report.report_id
        for report in reports
        if report.confidence == "sensor_confirmed" and report_truth[report.report_id]
    }


def _append_verification_actions(
    actions: list[CrisisopsAction],
    report: Report,
    report_truth: Mapping[str, bool],
    verified_report_ids: set[str],
    flagged_false_ids: set[str],
) -> None:
    if report.confidence != "sensor_confirmed" and report.report_id not in verified_report_ids:
        actions.append(
            _action(
                {
                    "type": "verify_report",
                    "report_id": report.report_id,
                    "verification_method": "cross_check",
                    "rationale": "Confirm non-sensor report before allocating scarce units.",
                }
            )
        )
        verified_report_ids.add(report.report_id)
    if report_truth[report.report_id] is False and report.report_id not in flagged_false_ids:
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
        flagged_false_ids.add(report.report_id)


def _append_noops_until(actions: list[CrisisopsAction], reveal_at_step: int) -> None:
    while len(actions) + 1 < reveal_at_step:
        actions.append(
            _action(
                {
                    "type": "noop",
                    "reason": "Wait for the next scheduled report reveal.",
                }
            )
        )


def _append_allocation_action(
    actions: list[CrisisopsAction],
    unit: Unit,
    zone: Zone,
    report_ids: list[str],
    assigned_units: set[str],
    allocated_unit_types_by_zone: dict[str, set[str]],
    resolved_zone_ids: set[str],
) -> None:
    assigned_units.add(unit.unit_id)
    allocated_unit_types_by_zone.setdefault(zone.zone_id, set()).add(unit.unit_type)
    resolved_zone_ids.add(zone.zone_id)
    actions.append(
        _action(
            {
                "type": "allocate_unit",
                "unit_id": unit.unit_id,
                "zone_id": zone.zone_id,
                "task": TASK_BY_UNIT_TYPE.get(unit.unit_type, "recon"),
                "priority": _priority_for_zone(zone),
                "report_ids": report_ids,
            }
        )
    )


def _next_unit_for_zone(
    units: Sequence[Unit], assigned_units: set[str], zone: Zone
) -> Unit | None:
    for unit_type in sorted(zone.required_unit_types):
        unit = _next_unit(units, assigned_units, unit_type)
        if unit is not None:
            return unit
    return None


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
    zones_by_id = hidden_state.get("zones_by_id", {})
    zone = zones_by_id.get(zone_id, {})
    return zone.get("access_status") == "blocked"


def _report_truth(report_id: str, hidden_state: Mapping[str, Any]) -> bool | None:
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
    for report_id in report_ids:
        report = reports_by_id.get(report_id)
        if not report:
            continue
        if report.get("confidence") != "sensor_confirmed" and report_id not in verified:
            return True
    return False


def _overfills_shelter(
    payload: Mapping[str, Any], hidden_state: Mapping[str, Any]
) -> bool:
    zone_id = payload.get("zone_id")
    shelter_id = payload.get("destination_shelter_id")
    if not zone_id or not shelter_id:
        return False
    zones_by_id = hidden_state.get("zones_by_id", {})
    zone = zones_by_id.get(zone_id, {})
    shelter = zone.get("shelter") or {}
    if shelter.get("shelter_id") != shelter_id:
        shelters_by_id = hidden_state.get("shelters_by_id", {})
        shelter = shelters_by_id.get(shelter_id, {})
    return bool(
        shelter
        and zone.get("population_at_risk", 0)
        > shelter.get("capacity_available", 0)
    )


def _next_unit(
    units: Sequence[Unit], assigned_units: set[str], unit_type: str
) -> Unit | None:
    return next(
        (
            unit
            for unit in units
            if unit.unit_id not in assigned_units
            and unit.unit_type == unit_type
            and unit.status == "available"
        ),
        None,
    )


def _is_critical(zone: Zone) -> bool:
    return zone.severity >= 4


def _priority_for_zone(zone: Zone) -> str:
    if zone.severity >= 5:
        return "critical"
    if zone.severity >= 4:
        return "high"
    if zone.severity <= 2:
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
