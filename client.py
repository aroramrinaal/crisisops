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
        >>> # Connect to a running server
        >>> with CrisisopsEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CrisisopsAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CrisisopsEnv.from_docker_image("crisisops-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CrisisopsAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CrisisopsAction) -> Dict:
        """
        Convert CrisisopsAction to JSON payload for step message.

        Args:
            action: CrisisopsAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

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
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
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
