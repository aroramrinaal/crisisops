"""Mid-episode event application for Crisisops scenarios."""

from collections.abc import Mapping, Sequence
from typing import Any

try:
    from ...models import Report, RouteInfo, Unit, Zone
except ImportError:
    from models import Report, RouteInfo, Unit, Zone


def apply_pending_events(
    time_step: int,
    zones: Sequence[Zone],
    reports: Sequence[Report],
    resources: Sequence[Unit],
    hidden_state: Mapping[str, Any],
) -> tuple[list[Zone], list[Report], list[Unit], list[str]]:
    """Apply deterministic scenario events scheduled for this time step."""

    updated_zones = [zone.model_copy(deep=True) for zone in zones]
    updated_reports = [report.model_copy(deep=True) for report in reports]
    updated_resources = [resource.model_copy(deep=True) for resource in resources]
    log_messages: list[str] = []

    for event in hidden_state.get("mid_episode_events", []):
        if event.get("time_step") != time_step:
            continue
        event_type = event.get("event_type")
        zone_id = event.get("zone_id")
        payload = event.get("payload", {})
        if event_type == "aftershock":
            updated_zones = _apply_aftershock(updated_zones, zone_id, payload)
            log_messages.append(f"Aftershock degraded access for {zone_id}.")
        elif event_type == "road_closure":
            updated_zones = _apply_road_closure(updated_zones, zone_id, payload)
            log_messages.append(f"Road closure blocked access near {zone_id}.")
        elif event_type == "shelter_overflow":
            updated_zones = _apply_shelter_overflow(updated_zones, zone_id)
            log_messages.append(f"Shelter capacity exhausted in {zone_id}.")
        elif event_type == "new_incident":
            updated_reports.append(_new_incident_report(time_step, zone_id, payload))
            log_messages.append(f"New incident report surfaced for {zone_id}.")
        elif event_type == "mutual_aid_unlock":
            updated_resources = _apply_mutual_aid_unlock(updated_resources, payload)
            log_messages.append("Mutual aid units are now available.")

    return updated_zones, updated_reports, updated_resources, log_messages


def _apply_aftershock(
    zones: list[Zone], zone_id: str, payload: Mapping[str, Any]
) -> list[Zone]:
    updated: list[Zone] = []
    for zone in zones:
        if zone.zone_id != zone_id:
            updated.append(zone)
            continue
        routes = [
            route.model_copy(
                update={
                    "status": "blocked",
                    "hazards": [*route.hazards, "aftershock"],
                }
            )
            for route in zone.routes
        ]
        updated.append(
            zone.model_copy(
                update={
                    "severity": min(5, zone.severity + int(payload.get("severity_delta", 1))),
                    "access_status": "blocked",
                    "routes": routes,
                }
            )
        )
    return updated


def _apply_road_closure(
    zones: list[Zone], zone_id: str, payload: Mapping[str, Any]
) -> list[Zone]:
    route_id = payload.get("route_id")
    updated: list[Zone] = []
    for zone in zones:
        if zone.zone_id != zone_id:
            updated.append(zone)
            continue
        routes = [
            _blocked_route(route)
            if route_id is None or route.route_id == route_id
            else route
            for route in zone.routes
        ]
        updated.append(
            zone.model_copy(update={"access_status": "degraded", "routes": routes})
        )
    return updated


def _apply_shelter_overflow(zones: list[Zone], zone_id: str) -> list[Zone]:
    updated: list[Zone] = []
    for zone in zones:
        if zone.zone_id == zone_id and zone.shelter is not None:
            updated.append(
                zone.model_copy(
                    update={
                        "shelter": zone.shelter.model_copy(
                            update={"status": "full", "capacity_available": 0}
                        )
                    }
                )
            )
        else:
            updated.append(zone)
    return updated


def _new_incident_report(
    time_step: int, zone_id: str, payload: Mapping[str, Any]
) -> Report:
    report_type = payload.get("report_type", "collapse")
    return Report(
        report_id=payload.get("report_id", f"report-event-{time_step}-{zone_id}"),
        zone_id=zone_id,
        source="official",
        report_type=report_type,
        severity="high",
        description=f"{report_type.replace('_', ' ')} reported during operations.",
        verified_status="unverified",
        confidence="official_unverified",
        time_step=time_step,
        reveal_at_step=time_step,
    )


def _apply_mutual_aid_unlock(
    resources: list[Unit], payload: Mapping[str, Any]
) -> list[Unit]:
    affected = set(payload.get("unit_ids", []))
    return [
        unit.model_copy(
            update={"status": "available", "current_zone_id": None, "district_id": None}
        )
        if unit.unit_id in affected
        else unit
        for unit in resources
    ]


def _blocked_route(route: RouteInfo) -> RouteInfo:
    return route.model_copy(
        update={"status": "blocked", "hazards": [*route.hazards, "road_closure"]}
    )
