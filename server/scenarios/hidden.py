"""Hidden truth assembly for deterministic Crisisops scenarios."""

from random import Random
from typing import Any

try:
    from ...models import Report, Unit, Zone
    from ..rules import priority_key
    from .config import INCIDENT_TYPES, TierConfig
except ImportError:
    from models import Report, Unit, Zone
    from server.rules import priority_key
    from server.scenarios.config import INCIDENT_TYPES, TierConfig


def build_hidden_truth(
    tier: str,
    seed: int,
    config: TierConfig,
    rng: Random,
    zones: list[Zone],
    reports: list[Report],
    units: list[Unit],
    report_truth: dict[str, bool],
) -> dict[str, Any]:
    mid_episode_events = _mid_episode_events(config, rng, zones, units)
    event_reports = _event_reports_from_events(mid_episode_events)
    all_reports = [*reports, *event_reports]
    report_truth_with_events = {
        **report_truth,
        **{report.report_id: True for report in event_reports},
    }
    true_zone_ids = {
        report.zone_id
        for report in all_reports
        if report_truth_with_events[report.report_id]
    }
    priority_zone_order = [
        zone.zone_id
        for zone in sorted(zones, key=priority_key)
        if zone.zone_id in true_zone_ids
    ]
    critical_zone_ids = [
        zone.zone_id
        for zone in zones
        if zone.zone_id in true_zone_ids and zone.severity >= 4
    ]
    first_degraded_district = "district-1" if config.comms_degradation else None
    sitrep_truth = _sitrep_truth(
        all_reports, report_truth_with_events, true_zone_ids
    )
    return {
        "tier": tier,
        "seed": seed,
        "config": config.__dict__,
        "report_truth": report_truth_with_events,
        "true_report_ids": sorted(
            report_id
            for report_id, is_true in report_truth_with_events.items()
            if is_true
        ),
        "false_report_ids": sorted(
            report_id
            for report_id, is_true in report_truth_with_events.items()
            if not is_true
        ),
        "incident_zones": sorted(true_zone_ids),
        "critical_zone_ids": sorted(critical_zone_ids),
        "priority_zone_order": priority_zone_order,
        "zone_incidents": _zone_incidents(
            all_reports, report_truth_with_events, true_zone_ids
        ),
        "zone_deadlines": {zone.zone_id: zone.deadline_steps for zone in zones},
        "zone_required_unit_types": {
            zone.zone_id: set(zone.required_unit_types) for zone in zones
        },
        "fatigue_enabled": config.fatigue,
        "mid_episode_events": mid_episode_events,
        "district_comms_degraded_from_step": 8 if config.comms_degradation else None,
        "degraded_district_id": first_degraded_district,
        "episode_cap": config.episode_cap,
        "verified_report_ids": [],
        "first_correct_allocation_steps": {},
        "deadline_missed_zone_ids": [],
        "reports_by_id": {
            report.report_id: report.model_dump(mode="json") for report in reports
        },
        "event_reports_by_id": {
            report.report_id: report.model_dump(mode="json")
            for report in event_reports
        },
        "zones_by_id": {zone.zone_id: zone.model_dump(mode="json") for zone in zones},
        "units_by_id": {unit.unit_id: unit.model_dump(mode="json") for unit in units},
        "shelters_by_id": {
            zone.shelter.shelter_id: zone.shelter.model_dump(mode="json")
            for zone in zones
            if zone.shelter is not None
        },
        "sitrep_truth": sitrep_truth,
    }


def _zone_incidents(
    reports: list[Report], report_truth: dict[str, bool], true_zone_ids: set[str]
) -> dict[str, list[str]]:
    return {
        zone_id: [
            report.report_type
            for report in reports
            if report.zone_id == zone_id and report_truth[report.report_id]
        ]
        for zone_id in sorted(true_zone_ids)
    }


def _mid_episode_events(
    config: TierConfig, rng: Random, zones: list[Zone], units: list[Unit]
) -> list[dict[str, Any]]:
    if not config.mid_episode_events:
        return []

    events: list[dict[str, Any]] = []
    target_zone = rng.choice(zones)
    events.append(
        {
            "time_step": min(6, max(2, config.episode_cap // 4)),
            "zone_id": target_zone.zone_id,
            "event_type": "aftershock",
            "payload": {"severity_delta": 1},
        }
    )
    road_zone = rng.choice([zone for zone in zones if zone.routes] or zones)
    route_id = road_zone.routes[0].route_id if road_zone.routes else None
    events.append(
        {
            "time_step": min(config.episode_cap - 2, max(3, config.episode_cap // 3)),
            "zone_id": road_zone.zone_id,
            "event_type": "road_closure",
            "payload": {"route_id": route_id},
        }
    )

    if config.district_count > 1:
        affected_units = [
            unit.unit_id for unit in units if unit.mutual_aid_unlock_step == 12
        ]
        events.append(
            {
                "time_step": 12,
                "zone_id": "",
                "event_type": "mutual_aid_unlock",
                "payload": {"unit_ids": affected_units},
            }
        )
        events.append(
            {
                "time_step": 14,
                "zone_id": rng.choice(zones).zone_id,
                "event_type": "new_incident",
                "payload": {
                    "report_id": "report-mid-1",
                    "report_type": rng.choice(INCIDENT_TYPES),
                },
            }
        )
    else:
        events.append(
            {
                "time_step": min(config.episode_cap - 1, 8),
                "zone_id": rng.choice(zones).zone_id,
                "event_type": "shelter_overflow",
                "payload": {},
            }
        )

    return sorted(events, key=lambda event: (event["time_step"], event["event_type"]))


def _sitrep_truth(
    reports: list[Report], report_truth: dict[str, bool], true_zone_ids: set[str]
) -> dict[str, list[str]]:
    true_report_ids = sorted(
        report.report_id for report in reports if report_truth[report.report_id]
    )
    false_report_ids = sorted(
        report.report_id for report in reports if not report_truth[report.report_id]
    )
    return {
        "incidents_confirmed": true_report_ids,
        "incidents_resolved": [],
        "unresolved_risks": sorted(true_zone_ids),
        "false_alarms_detected": false_report_ids,
    }


def _event_reports_from_events(events: list[dict[str, Any]]) -> list[Report]:
    reports: list[Report] = []
    for event in events:
        if event.get("event_type") != "new_incident":
            continue
        payload = event.get("payload", {})
        report_type = payload.get("report_type", "collapse")
        time_step = int(event["time_step"])
        reports.append(
            Report(
                report_id=payload.get(
                    "report_id", f"report-event-{time_step}-{event['zone_id']}"
                ),
                zone_id=event["zone_id"],
                source="official",
                report_type=report_type,
                severity="high",
                description=(
                    f"{report_type.replace('_', ' ')} reported during operations."
                ),
                verified_status="unverified",
                confidence="official_unverified",
                time_step=time_step,
                reveal_at_step=time_step,
            )
        )
    return reports
