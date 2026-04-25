"""Deterministic builders for visible Crisisops scenario objects."""

from random import Random

try:
    from ..models import Report, RouteInfo, ShelterInfo, Unit, Zone
    from ..rules import unit_type_for_incident
    from .config import REPORT_TYPES, RISK_BY_TRUTH, SOURCES, ZONE_NAMES, TierConfig
except ImportError:
    from models import Report, RouteInfo, ShelterInfo, Unit, Zone
    from server.rules import unit_type_for_incident
    from server.scenarios.config import (
        REPORT_TYPES,
        RISK_BY_TRUTH,
        SOURCES,
        ZONE_NAMES,
        TierConfig,
    )


def build_zones(config: TierConfig, rng: Random) -> list[Zone]:
    risks = ["low", "moderate", "high", "critical"]
    zones: list[Zone] = []
    for index in range(config.zone_count):
        zone_id = f"zone-{index + 1}"
        routes = _build_routes(index, zone_id, rng)
        shelter = _build_shelter(index, zone_id, rng)
        zones.append(
            Zone(
                zone_id=zone_id,
                name=ZONE_NAMES[index],
                risk_level=rng.choice(risks),
                population_estimate=rng.randint(800, 12000),
                infrastructure_status={
                    "power": rng.choice(["online", "degraded", "offline"]),
                    "roads": rng.choice(["online", "degraded", "offline"]),
                    "cell": rng.choice(["online", "degraded", "offline"]),
                },
                shelter=shelter,
                routes=routes,
            )
        )
    return zones


def build_reports(
    config: TierConfig, rng: Random, zones: list[Zone]
) -> tuple[list[Report], dict[str, bool]]:
    reports: list[Report] = []
    truth: dict[str, bool] = {}
    for index in range(config.report_count):
        report_id = f"report-{index + 1}"
        is_true = rng.random() >= 0.28
        zone = rng.choice(zones)
        report_type = rng.choice(REPORT_TYPES)
        severity = rng.choice(RISK_BY_TRUTH[is_true])
        reports.append(
            Report(
                report_id=report_id,
                zone_id=zone.zone_id,
                report_type=report_type,
                severity=severity,
                source=rng.choice(SOURCES),
                description=f"{report_type.replace('_', ' ')} reported near {zone.name}",
                verified_status="unverified",
                confidence=round(rng.uniform(0.35, 0.92), 2),
                time_step=rng.randint(0, max(0, config.episode_cap // 3)),
            )
        )
        truth[report_id] = is_true
    if reports and not any(truth.values()):
        truth[reports[0].report_id] = True
    return reports, truth


def build_units(
    config: TierConfig, reports: list[Report], report_truth: dict[str, bool]
) -> list[Unit]:
    needed_types = [
        unit_type_for_incident(report.report_type)
        for report in reports
        if report_truth[report.report_id]
    ]
    baseline_types = ["medical", "fire", "rescue", "police", "supply", "recon"]
    unit_types = list(dict.fromkeys(needed_types + baseline_types))
    budget = max(3, min(len(unit_types), config.zone_count + 1))
    return [
        Unit(
            unit_id=f"unit-{index + 1}",
            unit_type=unit_type,
            status="available",
            current_zone_id=None,
            capacity=1 if config.fatigue else 2,
            capabilities=[unit_type, "recon"] if unit_type != "recon" else ["recon"],
        )
        for index, unit_type in enumerate(unit_types[:budget])
    ]


def _build_shelter(index: int, zone_id: str, rng: Random) -> ShelterInfo:
    return ShelterInfo(
        shelter_id=f"shelter-{index + 1}",
        zone_id=zone_id,
        name=f"{ZONE_NAMES[index]} Relief School",
        status="closed",
        capacity_total=rng.randint(80, 260),
        capacity_available=rng.randint(30, 180),
        supplies={"water": rng.randint(20, 120), "medical_kits": rng.randint(2, 18)},
    )


def _build_routes(index: int, zone_id: str, rng: Random) -> list[RouteInfo]:
    if index == 0:
        return []
    return [
        RouteInfo(
            route_id=f"route-{index}-{index + 1}",
            from_zone_id=f"zone-{index}",
            to_zone_id=zone_id,
            status=rng.choice(["open", "open", "congested", "blocked"]),
            travel_time_minutes=rng.randint(8, 45),
            hazards=[],
        )
    ]
