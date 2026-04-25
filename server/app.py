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

import os

from fastapi.responses import RedirectResponse

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


SCAFFOLD_ROUTES = {
    "/",
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


# Create the app with web interface and README integration
app = create_app(
    CrisisopsEnvironment,
    CrisisopsAction,
    CrisisopsObservation,
    env_name="crisisops",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)
prune_scaffold_routes(app)
register_demo_ui(app)


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Send the root page to the custom demo UI."""
    return RedirectResponse(url="/demo", status_code=308)


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
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
