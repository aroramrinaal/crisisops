# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Crisisops Environment.

This module creates an HTTP server that exposes the CrisisopsEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Body, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# This project uses a hand-built HTML demo at /demo instead of OpenEnv's
# default Gradio web UI. openenv push may set this flag, so keep it disabled
# before creating the OpenEnv app.
os.environ["ENABLE_WEB_INTERFACE"] = "false"

try:
    from openenv.core.env_server import http_server as openenv_http_server
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e


def serialize_observation_with_metadata(observation):
    """Preserve Observation.metadata in OpenEnv response observation payloads."""
    obs_dict = observation.model_dump(exclude={"reward", "done"})
    return {
        "observation": obs_dict,
        "reward": observation.reward,
        "done": observation.done,
    }


openenv_http_server.serialize_observation = serialize_observation_with_metadata
create_app = openenv_http_server.create_app

try:
    from ..models import CrisisopsAction, CrisisopsObservation
    from .demo_ui import register_demo_ui
    from .crisisops_environment import CrisisopsEnvironment
except ImportError:
    from models import CrisisopsAction, CrisisopsObservation
    from server.demo_ui import register_demo_ui
    from server.crisisops_environment import CrisisopsEnvironment


@dataclass
class SessionRecord:
    env: CrisisopsEnvironment
    lock: asyncio.Lock


class SessionResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: Optional[int] = Field(default=None, ge=0)
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    episode_id: Optional[str] = None


class SessionStepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1)
    action: Dict[str, Any]


SCAFFOLD_ROUTES = {
    "/",
    "/reset",
    "/step",
    "/state",
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
    "/web",
}


def prune_scaffold_routes(app):
    """Remove scaffolded UI/documentation routes before adding the custom demo."""
    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) not in SCAFFOLD_ROUTES
    ]
    app.openapi_url = None
    app.docs_url = None
    app.redoc_url = None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    return value


def _serialize_observation(observation: CrisisopsObservation) -> Dict[str, Any]:
    return _json_safe(
        {
            "observation": observation.model_dump(exclude={"reward", "done"}),
            "reward": observation.reward,
            "done": observation.done,
        }
    )


# Create the app with web interface and README integration
app = create_app(
    CrisisopsEnvironment,
    CrisisopsAction,
    CrisisopsObservation,
    env_name="crisisops",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)
app.state.sessions = {}
app.state.sessions_lock = asyncio.Lock()
prune_scaffold_routes(app)
register_demo_ui(app)


async def _get_session(session_id: str) -> SessionRecord:
    async with app.state.sessions_lock:
        record = app.state.sessions.get(session_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown session_id: {session_id}",
        )
    return record


async def _close_all_sessions() -> None:
    async with app.state.sessions_lock:
        sessions = app.state.sessions
        app.state.sessions = {}
    for record in sessions.values():
        try:
            record.env.close()
        except Exception:
            pass


@app.on_event("shutdown")
async def shutdown_sessions() -> None:
    """Close in-memory HTTP sessions when the app shuts down."""

    await _close_all_sessions()


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Send the root page to the custom demo UI."""
    return RedirectResponse(url="/demo", status_code=308)


@app.post(
    "/reset",
    tags=["Environment Control"],
    summary="Reset the environment with a session-backed episode",
)
async def reset(
    request: SessionResetRequest = Body(default_factory=SessionResetRequest),
) -> Dict[str, Any]:
    """Create a new HTTP session and return the first observation."""

    env = CrisisopsEnvironment()
    session_id = request.episode_id or str(uuid4())
    task_id = request.task_id or request.task_name

    try:
        observation = env.reset(
            task_id=task_id,
            seed=request.seed,
            episode_id=session_id,
        )
    except Exception:
        env.close()
        raise

    record = SessionRecord(env=env, lock=asyncio.Lock())
    async with app.state.sessions_lock:
        app.state.sessions[session_id] = record

    payload = _serialize_observation(observation)
    payload["session_id"] = session_id
    return payload


@app.post(
    "/step",
    tags=["Environment Control"],
    summary="Execute an action inside an existing HTTP session",
)
async def step(request: SessionStepRequest) -> Dict[str, Any]:
    """Mutate only the environment associated with the provided session_id."""

    record = await _get_session(request.session_id)
    try:
        action = CrisisopsAction.model_validate(request.action)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc

    async with record.lock:
        observation = record.env.step(action)

    payload = _serialize_observation(observation)
    payload["session_id"] = request.session_id
    return payload


@app.get(
    "/state",
    tags=["State Management"],
    summary="Get state for an existing HTTP session",
)
async def state(session_id: Optional[str] = Query(default=None, min_length=1)) -> Dict[str, Any]:
    """Return state for the session that owns the episode."""

    if session_id is None:
        env = CrisisopsEnvironment()
        try:
            state_obj = env.state
            return _json_safe(
                state_obj.model_dump()
                if hasattr(state_obj, "model_dump")
                else dict(state_obj)
            )
        finally:
            env.close()

    record = await _get_session(session_id)
    async with record.lock:
        state_obj = record.env.state
        state_payload = _json_safe(
            state_obj.model_dump() if hasattr(state_obj, "model_dump") else dict(state_obj)
        )
    state_payload["session_id"] = session_id
    return state_payload


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m crisisops.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn crisisops.server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    resolved_port = args.port
    if resolved_port is None:
        resolved_port = int(os.environ.get("PORT", str(port)))
    uvicorn.run(app, host=args.host, port=resolved_port)


if __name__ == "__main__":
    main()
