"""Deterministic builders for visible Crisisops scenario objects."""

from random import Random

try:
    from ...models import Report, RouteInfo, ShelterInfo, Unit, Zone
    from ..rules import INCIDENT_UNIT_TYPES, unit_types_for_incident
    from .config import INCIDENT_TYPES, SOURCES, ZONE_NAMES, TierConfig
except ImportError:
    from models import Report, RouteInfo, ShelterInfo, Unit, Zone
    from server.rules import INCIDENT_UNIT_TYPES, unit_types_for_incident
    from server.scenarios.config import INCIDENT_TYPES, SOURCES, ZONE_NAMES, TierConfig


UNIT_TYPES = [
    "rescue_team",
    "medical_unit",
    "supply_truck",
    "evac_bus",
    "recon_drone",
]
CONFIDENCES = ["citizen", "official_unverified", "sensor_confirmed"]


def build_zones(config: TierConfig, rng: Random) -> list[Zone]:
    zones: list[Zone] = []
    incident_pool = list(INCIDENT_TYPES)
    for index in range(config.zone_count):
        zone_id = f"zone-{index + 1}"
        incident_type = incident_pool[index % len(incident_pool)]
        severity = rng.randint(1, 5)
        population = rng.randint(800, 15000)
        deadline_steps = _deadline_for_severity(severity, rng)
        district_id = (
            f"district-{(index % config.district_count) + 1}"
            if config.district_count > 1
            else None
        )
        zones.append(
            Zone(
                zone_id=zone_id,
                name=ZONE_NAMES[index],
                incident_type=incident_type,
                severity=severity,
                population_at_risk=population,
                deadline_steps=deadline_steps,
                access_status=rng.choices(
                    ["clear", "degraded", "blocked"],
                    weights=[0.68, 0.24, 0.08],
                    k=1,
                )[0],
                required_unit_types=unit_types_for_incident(incident_type),
                district_id=district_id,
                infrastructure_status={
                    "power": rng.choice(["online", "degraded", "offline"]),
                    "roads": rng.choice(["online", "degraded", "offline"]),
                    "cell": rng.choice(["online", "degraded", "offline"]),
                },
                shelter=_build_shelter(index, zone_id, population, rng),
                routes=_build_routes(index, zone_id, rng),
            )
        )
    return zones


def build_reports(
    config: TierConfig, rng: Random, zones: list[Zone]
) -> tuple[list[Report], dict[str, bool]]:
    reports: list[Report] = []
    truth: dict[str, bool] = {}
    truth_flags = [True] * config.true_report_count + [False] * config.false_report_count
    rng.shuffle(truth_flags)

    for index, is_true in enumerate(truth_flags):
        report_id = f"report-{index + 1}"
        zone = rng.choice(zones)
        report_type = zone.incident_type if is_true else _false_report_type(zone, rng)
        confidence = rng.choice(CONFIDENCES if is_true else CONFIDENCES[:2])
        reports.append(
            Report(
                report_id=report_id,
                zone_id=zone.zone_id,
                report_type=report_type,
                severity=_risk_for_severity(zone.severity if is_true else rng.randint(1, 4)),
                source=_source_for_confidence(confidence, rng),
                description=f"{report_type.replace('_', ' ')} reported near {zone.name}",
                verified_status="unverified",
                confidence=confidence,
                time_step=0,
                reveal_at_step=_reveal_step(config, index),
            )
        )
        truth[report_id] = is_true

    return reports, truth


def build_units(
    config: TierConfig, reports: list[Report], report_truth: dict[str, bool]
) -> list[Unit]:
    needed_types: list[str] = []
    for report in reports:
        if report_truth[report.report_id]:
            needed_types.extend(sorted(INCIDENT_UNIT_TYPES[report.report_type]))
    unit_types = list(dict.fromkeys(needed_types + UNIT_TYPES))
    while len(unit_types) < config.resource_count:
        unit_types.extend(UNIT_TYPES)

    units: list[Unit] = []
    for index, unit_type in enumerate(unit_types[: config.resource_count]):
        unlock_step = (
            12
            if config.district_count > 1 and index >= config.resource_count - 2
            else None
        )
        units.append(
            Unit(
                unit_id=f"unit-{index + 1}",
                unit_type=unit_type,
                status="offline" if unlock_step is not None else "available",
                current_zone_id="mutual-aid-staging" if unlock_step is not None else None,
                travel_cost=_travel_cost_for_unit(unit_type),
                fatigue=0,
                capacity=2 if unit_type == "evac_bus" else 1,
                capabilities=[unit_type],
                district_id=(
                    f"district-{(index % config.district_count) + 1}"
                    if config.district_count > 1
                    else None
                ),
                shared_pool=config.district_count > 1,
                mutual_aid_unlock_step=unlock_step,
            )
        )
    return units


def _deadline_for_severity(severity: int, rng: Random) -> int:
    ceiling = max(4, 13 - severity)
    floor = max(4, ceiling - 4)
    return rng.randint(floor, min(12, ceiling))


def _reveal_step(config: TierConfig, index: int) -> int:
    if not config.mid_episode_events:
        return 0
    stream_window = max(1, config.episode_cap // 2)
    return min(stream_window, (index * stream_window) // max(1, config.report_count - 1))


def _risk_for_severity(severity: int) -> str:
    if severity >= 5:
        return "critical"
    if severity == 4:
        return "high"
    if severity == 3:
        return "moderate"
    return "low"


def _source_for_confidence(confidence: str, rng: Random) -> str:
    if confidence == "sensor_confirmed":
        return "sensor"
    if confidence == "official_unverified":
        return "official"
    return rng.choice(["citizen", "field_team", "media"])


def _false_report_type(zone: Zone, rng: Random) -> str:
    choices = [
        incident_type
        for incident_type in INCIDENT_TYPES
        if incident_type != zone.incident_type
    ]
    return rng.choice(choices)


def _travel_cost_for_unit(unit_type: str) -> int:
    return {
        "recon_drone": 1,
        "rescue_team": 2,
        "medical_unit": 2,
        "supply_truck": 3,
        "evac_bus": 3,
    }[unit_type]


def _build_shelter(
    index: int, zone_id: str, population_at_risk: int, rng: Random
) -> ShelterInfo:
    capacity_total = rng.randint(population_at_risk, population_at_risk + 2500)
    return ShelterInfo(
        shelter_id=f"shelter-{index + 1}",
        zone_id=zone_id,
        name=f"{ZONE_NAMES[index]} Relief School",
        status="closed",
        capacity_total=capacity_total,
        capacity_available=rng.randint(population_at_risk, capacity_total),
        supplies={"water": rng.randint(120, 800), "medical_kits": rng.randint(12, 80)},
    )


def _build_routes(index: int, zone_id: str, rng: Random) -> list[RouteInfo]:
    if index == 0:
        return []
    return [
        RouteInfo(
            route_id=f"route-{index}-{index + 1}",
            from_zone_id=f"zone-{index}",
            to_zone_id=zone_id,
            status=rng.choices(["open", "congested", "blocked"], weights=[0.66, 0.24, 0.1], k=1)[0],
            travel_time_minutes=rng.randint(8, 45),
            hazards=[],
        )
    ]
