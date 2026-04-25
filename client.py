# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Crisisops Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CrisisopsAction, CrisisopsObservation


class CrisisopsEnv(
    EnvClient[CrisisopsAction, CrisisopsObservation, State]
):
    """
    Client for the Crisisops Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with CrisisopsEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(CrisisopsAction.model_validate({
        ...         "type": "noop",
        ...         "reason": "wait for verified reports",
        ...     }))
    """

    def _step_payload(self, action: CrisisopsAction) -> Dict:
        """
        Convert CrisisopsAction to JSON payload for step message.

        Args:
            action: CrisisopsAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[CrisisopsObservation]:
        """
        Parse server response into StepResult[CrisisopsObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CrisisopsObservation
        """
        obs_data = payload.get("observation", {})
        observation = CrisisopsObservation(
            visible_zones=obs_data.get("visible_zones", []),
            reports=obs_data.get("reports", []),
            resources=obs_data.get("resources", []),
            time_step=obs_data.get("time_step", 0),
            incident_log=obs_data.get("incident_log", []),
            session_id=obs_data.get("session_id", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", payload.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
