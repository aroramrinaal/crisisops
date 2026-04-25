"""Static demo UI registration for the Crisisops environment."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


DEMO_DIR = Path(__file__).resolve().parent / "demo"
ASSETS_DIR = DEMO_DIR / "assets"
INDEX_HTML = DEMO_DIR / "index.html"


def register_demo_ui(app: FastAPI) -> None:
    """Register the custom demo page and optional static assets."""
    if ASSETS_DIR.exists() and not any(
        getattr(route, "path", None) == "/demo/assets" for route in app.router.routes
    ):
        app.mount("/demo/assets", StaticFiles(directory=ASSETS_DIR), name="demo-assets")

    @app.get("/demo", include_in_schema=False)
    async def demo_page() -> FileResponse:
        return FileResponse(INDEX_HTML)
