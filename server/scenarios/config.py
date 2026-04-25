"""Tier configuration and content pools for Crisisops scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TierConfig:
    zone_count: int
    report_count: int
    true_report_count: int
    false_report_count: int
    resource_count: int
    fatigue: bool
    mid_episode_events: bool
    comms_degradation: bool
    episode_cap: int
    district_count: int = 1


TIER_CONFIGS = {
    "easy": TierConfig(
        zone_count=1,
        report_count=3,
        true_report_count=2,
        false_report_count=1,
        resource_count=2,
        fatigue=False,
        mid_episode_events=False,
        comms_degradation=False,
        episode_cap=8,
    ),
    "medium": TierConfig(
        zone_count=3,
        report_count=6,
        true_report_count=4,
        false_report_count=2,
        resource_count=4,
        fatigue=False,
        mid_episode_events=False,
        comms_degradation=False,
        episode_cap=15,
    ),
    "hard": TierConfig(
        zone_count=5,
        report_count=10,
        true_report_count=7,
        false_report_count=3,
        resource_count=6,
        fatigue=True,
        mid_episode_events=True,
        comms_degradation=False,
        episode_cap=25,
    ),
    "expert": TierConfig(
        zone_count=9,
        report_count=15,
        true_report_count=10,
        false_report_count=5,
        resource_count=8,
        fatigue=True,
        mid_episode_events=True,
        comms_degradation=True,
        episode_cap=40,
        district_count=3,
    ),
}

ZONE_NAMES = [
    "Riverside",
    "Old Market",
    "North Hospital",
    "Rail Yard",
    "Hill Sector",
    "Airport Road",
    "Industrial Belt",
    "Lake Ward",
    "South Substation",
]
INCIDENT_TYPES = [
    "flood",
    "collapse",
    "medical_surge",
    "fire",
    "contamination",
    "power_outage",
]
SOURCES = ["citizen", "sensor", "official", "field_team", "media"]
TRUE_REPORT_RATIO = {
    tier: config.true_report_count / config.report_count
    for tier, config in TIER_CONFIGS.items()
}
