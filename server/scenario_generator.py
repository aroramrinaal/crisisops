"""Public scenario generation entrypoint for Crisisops."""

from random import Random

try:
    from .rules import compute_optimal_plan
    from .scenarios.builders import build_reports, build_units, build_zones
    from .scenarios.config import TIER_CONFIGS
    from .scenarios.hidden import build_hidden_truth
except ImportError:
    from server.rules import compute_optimal_plan
    from server.scenarios.builders import build_reports, build_units, build_zones
    from server.scenarios.config import TIER_CONFIGS
    from server.scenarios.hidden import build_hidden_truth


def generate_scenario(tier: str, seed: int):
    """Return zones, units, reports, hidden truth, and computed optimal plan."""

    if tier not in TIER_CONFIGS:
        valid = ", ".join(sorted(TIER_CONFIGS))
        raise ValueError(f"Unknown tier '{tier}'. Expected one of: {valid}")

    rng = Random(seed)
    config = TIER_CONFIGS[tier]
    zones = build_zones(config, rng)
    reports, report_truth = build_reports(config, rng, zones)
    units = build_units(config, reports, report_truth)
    hidden_truth = build_hidden_truth(
        tier=tier,
        seed=seed,
        config=config,
        rng=rng,
        zones=zones,
        reports=reports,
        report_truth=report_truth,
    )
    scenario = {
        "zones": zones,
        "units": units,
        "reports": reports,
        "hidden_truth": hidden_truth,
    }
    optimal_plan = compute_optimal_plan(scenario)
    return zones, units, reports, hidden_truth, optimal_plan
