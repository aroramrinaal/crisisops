# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Crisisops Environment Implementation."""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CrisisopsAction, CrisisopsObservation
except ImportError:
    from models import CrisisopsAction, CrisisopsObservation


class CrisisopsEnvironment(Environment):
    """Minimal Phase 1 environment that emits the structured Crisisops contract."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the crisisops environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> CrisisopsObservation:
        """
        Reset the environment.

        Returns:
            Empty structured CrisisopsObservation for a new session.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return self._observation(["Crisisops environment ready."])

    def step(self, action: CrisisopsAction) -> CrisisopsObservation:  # type: ignore[override]
        """Accept a typed action and return the next structured observation."""
        self._state.step_count += 1

        action_payload = action.root
        action_type = action_payload.type
        return self._observation(
            [f"Accepted {action_type} action."],
            metadata={
                "action_type": action_type,
                "step": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def _observation(
        self, incident_log: list[str], metadata: dict | None = None
    ) -> CrisisopsObservation:
        return CrisisopsObservation(
            visible_zones=[],
            reports=[],
            resources=[],
            time_step=self._state.step_count,
            incident_log=incident_log,
            session_id=self._state.episode_id,
            done=False,
            reward=0.0,
            metadata=metadata or {},
        )
