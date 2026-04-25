# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Crisisops OpenEnv environment glue."""

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CrisisopsAction, CrisisopsObservation, Report, Unit, Zone
    from .grader import EasyGrader, ExpertGrader, HardGrader, MediumGrader
    from .reward import compute_step_reward
    from .scenario_generator import generate_scenario
    from .scenarios.events import apply_pending_events
except ImportError:
    from models import CrisisopsAction, CrisisopsObservation, Report, Unit, Zone
    from server.grader import EasyGrader, ExpertGrader, HardGrader, MediumGrader
    from server.reward import compute_step_reward
    from server.scenario_generator import generate_scenario
    from server.scenarios.events import apply_pending_events


TASK_TIERS = {
    "single_zone_response": "easy",
    "multi_zone_triage": "medium",
    "cascading_crisis": "hard",
    "multi_district_coordination": "expert",
}
GRADERS = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
    "expert": ExpertGrader,
}
DEFAULT_TASK_ID = "single_zone_response"


class CrisisopsEnvironment(Environment):
    """Stateful crisis-command simulation wired to OpenEnv interfaces."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
            current_task_id=DEFAULT_TASK_ID,
        )
        self._current_task_id = DEFAULT_TASK_ID
        self._tier = TASK_TIERS[DEFAULT_TASK_ID]
        self._zones: list[Zone] = []
        self._reports: list[Report] = []
        self._resources: list[Unit] = []
        self.hidden_state: dict[str, Any] = {}
        self.optimal_plan: list[CrisisopsAction] = []
        self.grader = EasyGrader()
        self.action_log: list[CrisisopsAction] = []
        self._runtime_state: dict[str, Any] = self._empty_runtime_state()

    def reset(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> CrisisopsObservation:
        """Start a deterministic task episode and return the visible observation."""

        self._current_task_id = task_id or DEFAULT_TASK_ID
        self._tier = self._tier_for_task(self._current_task_id)
        scenario_seed = seed if seed is not None else 42
        episode = episode_id or str(uuid4())

        (
            self._zones,
            self._resources,
            self._reports,
            self.hidden_state,
            self.optimal_plan,
        ) = generate_scenario(self._tier, scenario_seed)
        self.grader = GRADERS[self._tier]()
        self.action_log = []
        self._runtime_state = self._empty_runtime_state()
        self._state = State(
            episode_id=episode,
            step_count=0,
            current_task_id=self._current_task_id,
        )

        return self._observation(
            reward=0.0,
            done=False,
            incident_log=[
                f"Task {self._current_task_id} initialized at tier {self._tier}."
            ],
            metadata={
                "task_id": self._current_task_id,
                "tier": self._tier,
                "episode_cap": self.hidden_state["episode_cap"],
            },
        )

    def step(self, action: CrisisopsAction) -> CrisisopsObservation:  # type: ignore[override]
        """Apply an action, update visible state, and return reward-bearing observation."""

        if not self.hidden_state:
            self.reset(task_id=self._current_task_id)

        prev_runtime_state = self._runtime_snapshot()
        payload = action.root.model_dump(mode="json")

        self._state.step_count += 1
        (
            self._zones,
            self._reports,
            self._resources,
            event_log,
        ) = apply_pending_events(
            self._state.step_count,
            self._zones,
            self._reports,
            self._resources,
            self.hidden_state,
        )
        if event_log:
            self._refresh_hidden_indexes()
        self._apply_action(payload)
        self.action_log.append(action)
        self.hidden_state["verified_report_ids"] = list(
            self._runtime_state["verified_report_ids"]
        )
        self._update_deadline_misses()
        new_runtime_state = self._runtime_snapshot()

        reward = compute_step_reward(
            action,
            prev_runtime_state,
            new_runtime_state,
            self.hidden_state,
        )
        done = self._is_done(payload)
        metadata = {
            "task_id": self._current_task_id,
            "tier": self._tier,
            "action_type": payload["type"],
            "step": self._state.step_count,
        }
        if done:
            metadata["terminal_score"] = self.grader.grade(
                self.action_log, self.hidden_state
            )

        return self._observation(
            reward=reward,
            done=done,
            incident_log=[*event_log, self._incident_message(payload, done)],
            metadata=metadata,
        )

    @property
    def state(self) -> State:
        """Expose OpenEnv state plus the active task id."""

        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            current_task_id=self._current_task_id,
        )

    def _apply_action(self, payload: Mapping[str, Any]) -> None:
        action_type = payload["type"]
        if action_type == "verify_report":
            self._verify_report(payload["report_id"])
        elif action_type == "flag_false_alarm":
            self._runtime_state["false_alarm_report_ids"].add(payload["report_id"])
            self._set_report_status(payload["report_id"], "false_alarm")
        elif action_type == "allocate_unit":
            unit_id = payload["unit_id"]
            self._runtime_state["allocated_unit_ids"].add(unit_id)
            self._assign_unit(unit_id, payload["zone_id"])
            self._track_correct_allocation(unit_id, payload["zone_id"])
        elif action_type == "reroute_unit":
            self._runtime_state["rerouted_unit_ids"].add(payload["unit_id"])
        elif action_type == "issue_evacuation":
            self._runtime_state["evacuated_zone_ids"].add(payload["zone_id"])
        elif action_type == "open_shelter":
            shelter = payload["shelter"]
            self._runtime_state["opened_shelter_ids"].add(shelter["shelter_id"])
            self._open_shelter(shelter["shelter_id"])
        elif action_type == "dispatch_supplies":
            self._runtime_state["supplied_zone_ids"].add(
                payload["destination_zone_id"]
            )
        elif action_type == "request_recon":
            self._runtime_state["recon_zone_ids"].add(payload["zone_id"])
        elif action_type == "publish_sitrep":
            self._runtime_state["published_sitrep"] = True
            self._runtime_state["latest_sitrep"] = dict(payload["payload"])

        if action_type == "noop":
            self._runtime_state["consecutive_noop_count"] += 1
        else:
            self._runtime_state["consecutive_noop_count"] = 0

    def _verify_report(self, report_id: str) -> None:
        self._runtime_state["verified_report_ids"].add(report_id)
        is_true = self.hidden_state["report_truth"].get(
            report_id, self.hidden_state.get("event_report_truth", {}).get(report_id)
        )
        self._set_report_status(report_id, "verified" if is_true else "disputed")

    def _set_report_status(self, report_id: str, status: str) -> None:
        self._reports = [
            report.model_copy(update={"verified_status": status})
            if report.report_id == report_id
            else report
            for report in self._reports
        ]

    def _assign_unit(self, unit_id: str, zone_id: str) -> None:
        self._resources = [
            unit.model_copy(
                update={
                    "status": "assigned",
                    "current_zone_id": zone_id,
                    "fatigue": unit.fatigue + 1,
                }
            )
            if unit.unit_id == unit_id
            else unit
            for unit in self._resources
        ]

    def _track_correct_allocation(self, unit_id: str, zone_id: str) -> None:
        unit = next((unit for unit in self._resources if unit.unit_id == unit_id), None)
        zone = next((zone for zone in self._zones if zone.zone_id == zone_id), None)
        if unit is None or zone is None:
            return
        if unit.unit_type not in zone.required_unit_types:
            return
        steps = self._runtime_state["first_correct_allocation_steps"]
        steps.setdefault(zone_id, self._state.step_count)
        self.hidden_state["first_correct_allocation_steps"] = dict(steps)

    def _open_shelter(self, shelter_id: str) -> None:
        updated_zones = []
        for zone in self._zones:
            if zone.shelter is not None and zone.shelter.shelter_id == shelter_id:
                updated_zones.append(
                    zone.model_copy(
                        update={"shelter": zone.shelter.model_copy(update={"status": "open"})}
                    )
                )
            else:
                updated_zones.append(zone)
        self._zones = updated_zones

    def _is_done(self, payload: Mapping[str, Any]) -> bool:
        return (
            payload["type"] == "publish_sitrep"
            or self._state.step_count >= self.hidden_state["episode_cap"]
        )

    def _observation(
        self,
        reward: float,
        done: bool,
        incident_log: list[str],
        metadata: dict[str, Any],
    ) -> CrisisopsObservation:
        visible_reports = self._visible_reports()
        return CrisisopsObservation(
            visible_zones=self._zones,
            reports=visible_reports,
            resources=self._resources,
            time_step=self._state.step_count,
            incident_log=incident_log,
            session_id=self._state.episode_id or "",
            done=done,
            reward=reward,
            metadata=metadata,
        )

    def _runtime_snapshot(self) -> dict[str, Any]:
        return {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in self._runtime_state.items()
        }

    def _empty_runtime_state(self) -> dict[str, Any]:
        return {
            "verified_report_ids": set(),
            "allocated_unit_ids": set(),
            "false_alarm_report_ids": set(),
            "rerouted_unit_ids": set(),
            "evacuated_zone_ids": set(),
            "opened_shelter_ids": set(),
            "supplied_zone_ids": set(),
            "recon_zone_ids": set(),
            "published_sitrep": False,
            "latest_sitrep": None,
            "first_correct_allocation_steps": {},
            "consecutive_noop_count": 0,
            "deadline_missed_zone_ids": set(),
        }

    def _visible_reports(self) -> list[Report]:
        reports = [
            report
            for report in self._reports
            if report.reveal_at_step <= self._state.step_count
        ]
        if not self._expert_comms_degraded():
            return reports
        degraded_district = self.hidden_state.get("degraded_district_id")
        zone_districts = {zone.zone_id: zone.district_id for zone in self._zones}
        visible: list[Report] = []
        for report in reports:
            if zone_districts.get(report.zone_id) != degraded_district:
                visible.append(report)
                continue
            lowered_confidence = (
                "official_unverified"
                if report.confidence == "sensor_confirmed"
                else "citizen"
            )
            visible.append(
                report.model_copy(
                    update={
                        "confidence": lowered_confidence,
                        "verified_status": (
                            "unverified"
                            if report.verified_status == "verified"
                            else report.verified_status
                        ),
                    }
                )
            )
        return visible

    def _expert_comms_degraded(self) -> bool:
        degraded_from = self.hidden_state.get("district_comms_degraded_from_step")
        return bool(degraded_from is not None and self._state.step_count >= degraded_from)

    def _update_deadline_misses(self) -> None:
        first_correct = self._runtime_state["first_correct_allocation_steps"]
        missed = self._runtime_state["deadline_missed_zone_ids"]
        for zone_id in self.hidden_state.get("critical_zone_ids", []):
            deadline = self.hidden_state.get("zone_deadlines", {}).get(zone_id)
            if deadline is None or self._state.step_count < deadline:
                continue
            if first_correct.get(zone_id, deadline + 1) < deadline:
                continue
            missed.add(zone_id)
        self.hidden_state["deadline_missed_zone_ids"] = sorted(missed)

    def _refresh_hidden_indexes(self) -> None:
        self.hidden_state["reports_by_id"] = {
            report.report_id: report.model_dump(mode="json") for report in self._reports
        }
        self.hidden_state["zones_by_id"] = {
            zone.zone_id: zone.model_dump(mode="json") for zone in self._zones
        }
        self.hidden_state["units_by_id"] = {
            unit.unit_id: unit.model_dump(mode="json") for unit in self._resources
        }
        self.hidden_state["shelters_by_id"] = {
            zone.shelter.shelter_id: zone.shelter.model_dump(mode="json")
            for zone in self._zones
            if zone.shelter is not None
        }

    def _incident_message(self, payload: Mapping[str, Any], done: bool) -> str:
        suffix = " Episode complete." if done else ""
        return f"Accepted {payload['type']} action at step {self._state.step_count}.{suffix}"

    def _tier_for_task(self, task_id: str) -> str:
        if task_id in TASK_TIERS:
            return TASK_TIERS[task_id]
        if task_id in GRADERS:
            return task_id
        return TASK_TIERS[DEFAULT_TASK_ID]
