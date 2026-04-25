"""Tier configuration and content pools for Crisisops scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TierConfig:
    zone_count: int
    report_count: int
    fatigue: bool
    mid_episode_events: bool
    comms_degradation: bool
    episode_cap: int


TIER_CONFIGS = {
    "easy": TierConfig(
        zone_count=2,
        report_count=3,
        fatigue=False,
        mid_episode_events=False,
        comms_degradation=False,
        episode_cap=8,
    ),
    "medium": TierConfig(
        zone_count=4,
        report_count=6,
        fatigue=True,
        mid_episode_events=False,
        comms_degradation=False,
        episode_cap=12,
    ),
    "hard": TierConfig(
        zone_count=6,
        report_count=10,
        fatigue=True,
        mid_episode_events=True,
        comms_degradation=True,
        episode_cap=16,
    ),
    "expert": TierConfig(
        zone_count=8,
        report_count=14,
        fatigue=True,
        mid_episode_events=True,
        comms_degradation=True,
        episode_cap=22,
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
]
REPORT_TYPES = [
    "flood",
    "fire",
    "collapse",
    "medical",
    "infrastructure",
    "shelter",
    "resource",
    "other",
]
SOURCES = ["citizen", "sensor", "official", "field_team", "media"]
RISK_BY_TRUTH = {
    True: ["moderate", "high", "critical"],
    False: ["low", "moderate", "high"],
}
