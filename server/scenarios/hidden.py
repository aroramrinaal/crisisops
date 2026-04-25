"""Hidden truth assembly for deterministic Crisisops scenarios."""

from random import Random
from typing import Any

try:
    from ..models import Report, Zone
    from ..rules import priority_key
    from .config import TierConfig
except ImportError:
    from models import Report, Zone
    from server.rules import priority_key
    from server.scenarios.config import TierConfig


def build_hidden_truth(
    tier: str,
    seed: int,
    config: TierConfig,
    rng: Random,
    zones: list[Zone],
    reports: list[Report],
    report_truth: dict[str, bool],
) -> dict[str, Any]:
    true_zone_ids = {
        report.zone_id for report in reports if report_truth[report.report_id]
    }
    priority_zone_order = [
        zone.zone_id
        for zone in sorted(zones, key=priority_key)
        if zone.zone_id in true_zone_ids
    ]
    return {
        "tier": tier,
        "seed": seed,
        "config": config.__dict__,
        "report_truth": report_truth,
        "true_report_ids": sorted(
            report_id for report_id, is_true in report_truth.items() if is_true
        ),
        "false_report_ids": sorted(
            report_id for report_id, is_true in report_truth.items() if not is_true
        ),
        "incident_zones": sorted(true_zone_ids),
        "priority_zone_order": priority_zone_order,
        "zone_incidents": _zone_incidents(reports, report_truth, true_zone_ids),
        "fatigue_enabled": config.fatigue,
        "mid_episode_events": _mid_episode_events(config, rng, zones),
        "comms_degradation": _comms_degradation(config, rng, zones),
        "episode_cap": config.episode_cap,
        "verified_report_ids": [],
        "reports_by_id": {
            report.report_id: report.model_dump(mode="json") for report in reports
        },
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
    config: TierConfig, rng: Random, zones: list[Zone]
) -> list[dict[str, Any]]:
    if not config.mid_episode_events:
        return []
    return [
        {
            "time_step": rng.randint(2, config.episode_cap - 2),
            "zone_id": rng.choice(zones).zone_id,
            "event": rng.choice(["aftershock", "road_closure", "shelter_overflow"]),
        }
        for _ in range(2)
    ]


def _comms_degradation(
    config: TierConfig, rng: Random, zones: list[Zone]
) -> dict[str, float]:
    if not config.comms_degradation:
        return {}
    return {
        zone.zone_id: round(rng.uniform(0.1, 0.45), 2)
        for zone in zones
        if rng.random() < 0.5
    }
