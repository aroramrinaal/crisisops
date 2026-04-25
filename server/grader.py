"""Episode-level Crisisops graders."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from ..models import CrisisopsAction
    from .rules import compute_optimal_plan, is_unsafe_action
except ImportError:
    from models import CrisisopsAction
    from server.rules import compute_optimal_plan, is_unsafe_action


@dataclass(frozen=True)
class GradeBreakdown:
    verification: float
    allocation: float
    prioritization: float
    safety: float
    adaptation: float


class BaseGrader:
    """Shared grader that scores complete action logs against hidden truth."""

    weights = {
        "verification": 0.25,
        "allocation": 0.25,
        "prioritization": 0.2,
        "safety": 0.2,
        "adaptation": 0.1,
    }

    def grade(
        self,
        action_log: Sequence[CrisisopsAction | Mapping[str, Any]],
        hidden_state: Mapping[str, Any],
    ) -> float:
        """Return clamped episode score in [0.01, 0.99]."""

        actions = [_action_payload(action) for action in action_log]
        breakdown = self.subscores(actions, hidden_state)
        weighted = sum(
            getattr(breakdown, name) * weight for name, weight in self.weights.items()
        )
        return max(0.01, min(0.99, float(weighted)))

    def subscores(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> GradeBreakdown:
        return GradeBreakdown(
            verification=self._verification_score(actions, hidden_state),
            allocation=self._allocation_score(actions, hidden_state),
            prioritization=self._prioritization_score(actions, hidden_state),
            safety=self._safety_score(actions, hidden_state),
            adaptation=self._adaptation_score(actions, hidden_state),
        )

    def _verification_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        report_ids = set(hidden_state.get("report_truth", {}))
        if not report_ids:
            return 1.0
        verified_ids = {
            action.get("report_id")
            for action in actions
            if action.get("type") == "verify_report"
        }
        false_alarm_ids = {
            action.get("report_id")
            for action in actions
            if action.get("type") == "flag_false_alarm"
        }
        false_ids = set(hidden_state.get("false_report_ids", []))
        verification_rate = len(report_ids & verified_ids) / len(report_ids)
        false_alarm_rate = (
            len(false_ids & false_alarm_ids) / len(false_ids) if false_ids else 1.0
        )
        return (0.7 * verification_rate) + (0.3 * false_alarm_rate)

    def _allocation_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        required_zones = set(hidden_state.get("incident_zones", []))
        if not required_zones:
            return 1.0
        allocated_zones = {
            action.get("zone_id")
            for action in actions
            if action.get("type") in {"allocate_unit", "issue_evacuation"}
        }
        supplied_zones = {
            action.get("destination_zone_id")
            for action in actions
            if action.get("type") == "dispatch_supplies"
        }
        covered = required_zones & (allocated_zones | supplied_zones)
        return len(covered) / len(required_zones)

    def _prioritization_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        priority_zones = list(
            hidden_state.get("priority_zone_order")
            or hidden_state.get("incident_zones", [])
        )
        if not priority_zones:
            return 1.0
        first_operational_zone = next(
            (
                action.get("zone_id") or action.get("destination_zone_id")
                for action in actions
                if action.get("type")
                in {"allocate_unit", "issue_evacuation", "dispatch_supplies"}
            ),
            None,
        )
        if first_operational_zone in priority_zones[:1]:
            return 1.0
        if first_operational_zone in priority_zones:
            return 0.65
        return 0.25

    def _safety_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        if not actions:
            return 0.5
        unsafe_count = sum(1 for action in actions if is_unsafe_action(action, hidden_state))
        return max(0.0, 1.0 - (unsafe_count / len(actions)))

    def _adaptation_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        mid_events = hidden_state.get("mid_episode_events", [])
        comms = hidden_state.get("comms_degradation", {})
        if not mid_events and not comms:
            return 1.0
        recon_zones = {
            action.get("zone_id")
            for action in actions
            if action.get("type") in {"request_recon", "reroute_unit"}
        }
        event_zones = {event.get("zone_id") for event in mid_events}
        degraded_zones = set(comms)
        target_zones = event_zones | degraded_zones
        if not target_zones:
            return 1.0
        return len(target_zones & recon_zones) / len(target_zones)

    def grade_scenario_plan(self, scenario: Mapping[str, Any]) -> float:
        """Convenience hook for validating the derived optimal plan."""

        return self.grade(compute_optimal_plan(scenario), scenario["hidden_truth"])


class EasyGrader(BaseGrader):
    weights = {
        "verification": 0.3,
        "allocation": 0.25,
        "prioritization": 0.15,
        "safety": 0.2,
        "adaptation": 0.1,
    }


class MediumGrader(BaseGrader):
    weights = {
        "verification": 0.25,
        "allocation": 0.25,
        "prioritization": 0.2,
        "safety": 0.2,
        "adaptation": 0.1,
    }


class HardGrader(BaseGrader):
    weights = {
        "verification": 0.22,
        "allocation": 0.24,
        "prioritization": 0.2,
        "safety": 0.22,
        "adaptation": 0.12,
    }


class ExpertGrader(BaseGrader):
    weights = {
        "verification": 0.2,
        "allocation": 0.22,
        "prioritization": 0.2,
        "safety": 0.23,
        "adaptation": 0.15,
    }


def _action_payload(action: CrisisopsAction | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, CrisisopsAction):
        return action.root.model_dump(mode="json")
    if "root" in action and isinstance(action["root"], Mapping):
        return dict(action["root"])
    return dict(action)
