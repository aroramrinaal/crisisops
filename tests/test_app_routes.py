"""Route tests for the FastAPI app wiring."""

from importlib import import_module
import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient


def _load_app(enable_web_interface: bool):
    module_name = "server.app"
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    sys.modules.pop(module_name, None)
    os.environ["ENABLE_WEB_INTERFACE"] = "true" if enable_web_interface else "false"
    return import_module(module_name).app


class TestAppRoutes:
    def test_root_redirects_to_demo_when_web_interface_enabled(self):
        client = TestClient(_load_app(enable_web_interface=True))

        response = client.get("/", follow_redirects=False)

        assert response.status_code == 308
        assert response.headers["location"] == "/demo"

    def test_demo_and_core_routes_are_available(self):
        client = TestClient(_load_app(enable_web_interface=False))

        demo_response = client.get("/demo")
        health_response = client.get("/health")
        reset_response = client.post("/reset")
        step_response = client.post("/step", json={"action": {"message": "hello"}})
        state_response = client.get("/state")
        schema_response = client.get("/schema")

        assert demo_response.status_code == 200
        assert health_response.status_code == 200
        assert reset_response.status_code == 200
        assert reset_response.json()["observation"]["echoed_message"] == (
            "Crisisops environment ready!"
        )
        assert step_response.status_code == 200
        assert step_response.json()["observation"]["echoed_message"] == "hello"
        assert state_response.status_code == 200
        assert schema_response.status_code == 200

    def test_scaffold_documentation_and_web_routes_are_removed(self):
        client = TestClient(_load_app(enable_web_interface=True))

        assert client.get("/web").status_code == 404
        assert client.get("/docs").status_code == 404
        assert client.get("/docs/oauth2-redirect").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_step_rejects_missing_action_payload(self):
        client = TestClient(_load_app(enable_web_interface=False))

        response = client.post("/step")

        assert response.status_code == 422
