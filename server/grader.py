"""Episode-level Crisisops graders."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from ..models import CrisisopsAction
    from .rules import INCIDENT_UNIT_TYPES, compute_optimal_plan, is_unsafe_action
except ImportError:
    from models import CrisisopsAction
    from server.rules import INCIDENT_UNIT_TYPES, compute_optimal_plan, is_unsafe_action


@dataclass(frozen=True)
class GradeBreakdown:
    incident_understanding: float
    resource_allocation: float
    safety: float
    timeliness: float
    sitrep_quality: float


class BaseGrader:
    """Shared grader that scores complete action logs against hidden truth."""

    weights = {
        "incident_understanding": 0.25,
        "resource_allocation": 0.30,
        "safety": 0.20,
        "timeliness": 0.15,
        "sitrep_quality": 0.10,
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
            incident_understanding=self._incident_understanding_score(
                actions, hidden_state
            ),
            resource_allocation=self._resource_allocation_score(actions, hidden_state),
            safety=self._safety_score(actions, hidden_state),
            timeliness=self._timeliness_score(actions, hidden_state),
            sitrep_quality=self._sitrep_quality_score(actions, hidden_state),
        )

    def _incident_understanding_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        verified_true = {
            action.get("report_id")
            for action in actions
            if action.get("type") == "verify_report"
            and hidden_state.get("report_truth", {}).get(action.get("report_id")) is True
        }
        verified_true |= {
            report_id
            for report_id in hidden_state.get("true_report_ids", [])
            if hidden_state.get("reports_by_id", {})
            .get(report_id, {})
            .get("confidence")
            == "sensor_confirmed"
        }
        flagged_false = {
            action.get("report_id")
            for action in actions
            if action.get("type") == "flag_false_alarm"
        }
        true_score = _f1(verified_true, set(hidden_state.get("true_report_ids", [])))
        false_score = _f1(flagged_false, set(hidden_state.get("false_report_ids", [])))
        return (true_score + false_score) / 2

    def _resource_allocation_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        allocate_actions = [
            action for action in actions if action.get("type") == "allocate_unit"
        ]
        if not allocate_actions:
            return 0.0
        priority_order = list(hidden_state.get("priority_zone_order", []))
        max_rank = max(1, len(priority_order))
        units_by_id = hidden_state.get("units_by_id", {})
        zones_by_id = hidden_state.get("zones_by_id", {})
        earned = 0.0
        possible = 0.0
        for action in allocate_actions:
            zone_id = action.get("zone_id")
            zone = zones_by_id.get(zone_id, {})
            unit = units_by_id.get(action.get("unit_id"), {})
            rank = priority_order.index(zone_id) if zone_id in priority_order else max_rank
            weight = max(1, max_rank - rank)
            possible += weight
            if unit.get("unit_type") in INCIDENT_UNIT_TYPES.get(
                zone.get("incident_type"), set()
            ):
                earned += weight
        return earned / possible if possible else 0.0

    def _safety_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        if not actions:
            return 0.5
        unsafe_count = 0
        verified: set[str] = set(hidden_state.get("verified_report_ids", []))
        for action in actions:
            running_hidden = {**hidden_state, "verified_report_ids": sorted(verified)}
            if is_unsafe_action(action, running_hidden):
                unsafe_count += 1
            if action.get("type") == "verify_report":
                verified.add(action.get("report_id"))
        return max(0.0, 1.0 - (unsafe_count / len(actions)))

    def _timeliness_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        critical_zones = set(hidden_state.get("critical_zone_ids", []))
        if not critical_zones:
            return 1.0
        first_correct = dict(hidden_state.get("first_correct_allocation_steps", {}))
        if not first_correct:
            first_correct = self._first_correct_steps_from_actions(actions, hidden_state)
        on_time = 0
        deadlines = hidden_state.get("zone_deadlines", {})
        for zone_id in critical_zones:
            deadline = deadlines.get(zone_id)
            if deadline is not None and first_correct.get(zone_id, deadline + 1) < deadline:
                on_time += 1
        return on_time / len(critical_zones)

    def _sitrep_quality_score(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> float:
        sitrep = next(
            (
                action.get("payload", {})
                for action in reversed(actions)
                if action.get("type") == "publish_sitrep"
            ),
            None,
        )
        if not sitrep:
            return 0.0
        first_correct = dict(hidden_state.get("first_correct_allocation_steps", {}))
        if not first_correct:
            first_correct = self._first_correct_steps_from_actions(actions, hidden_state)
        truth = _sitrep_truth_with_runtime(hidden_state, first_correct)
        fields = [
            "incidents_confirmed",
            "incidents_resolved",
            "unresolved_risks",
            "false_alarms_detected",
        ]
        scores = [
            _f1(set(sitrep.get(field, [])), set(truth.get(field, [])))
            for field in fields
        ]
        return sum(scores) / len(scores)

    def _first_correct_steps_from_actions(
        self, actions: Sequence[Mapping[str, Any]], hidden_state: Mapping[str, Any]
    ) -> dict[str, int]:
        units_by_id = hidden_state.get("units_by_id", {})
        zones_by_id = hidden_state.get("zones_by_id", {})
        first: dict[str, int] = {}
        for step, action in enumerate(actions, start=1):
            if action.get("type") != "allocate_unit":
                continue
            zone_id = action.get("zone_id")
            zone = zones_by_id.get(zone_id, {})
            unit = units_by_id.get(action.get("unit_id"), {})
            if unit.get("unit_type") in INCIDENT_UNIT_TYPES.get(
                zone.get("incident_type"), set()
            ):
                first.setdefault(zone_id, step)
        return first

    def grade_scenario_plan(self, scenario: Mapping[str, Any]) -> float:
        """Convenience hook for validating the derived optimal plan."""

        return self.grade(compute_optimal_plan(scenario), scenario["hidden_truth"])


class EasyGrader(BaseGrader):
    weights = {
        "incident_understanding": 0.28,
        "resource_allocation": 0.28,
        "safety": 0.22,
        "timeliness": 0.12,
        "sitrep_quality": 0.10,
    }


class MediumGrader(BaseGrader):
    weights = BaseGrader.weights


class HardGrader(BaseGrader):
    weights = {
        "incident_understanding": 0.23,
        "resource_allocation": 0.30,
        "safety": 0.20,
        "timeliness": 0.17,
        "sitrep_quality": 0.10,
    }


class ExpertGrader(BaseGrader):
    weights = {
        "incident_understanding": 0.22,
        "resource_allocation": 0.30,
        "safety": 0.20,
        "timeliness": 0.18,
        "sitrep_quality": 0.10,
    }


def _sitrep_truth_with_runtime(
    hidden_state: Mapping[str, Any], first_correct: Mapping[str, int]
) -> dict[str, list[str]]:
    truth = {
        key: list(value)
        for key, value in hidden_state.get("sitrep_truth", {}).items()
    }
    resolved = sorted(first_correct)
    incident_zones = set(hidden_state.get("incident_zones", []))
    truth["incidents_resolved"] = resolved
    truth["unresolved_risks"] = sorted(incident_zones - set(resolved))
    return truth


def _f1(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    true_positive = len(predicted & expected)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    return (2 * precision * recall) / (precision + recall)


def _action_payload(action: CrisisopsAction | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, CrisisopsAction):
        return action.root.model_dump(mode="json")
    if "root" in action and isinstance(action["root"], Mapping):
        return dict(action["root"])
    return dict(action)
