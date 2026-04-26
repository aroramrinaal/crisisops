"""Static demo UI registration for the Crisisops environment."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


DEMO_DIR = Path(__file__).resolve().parent / "demo"
ASSETS_DIR = DEMO_DIR / "assets"
INDEX_HTML = DEMO_DIR / "index.html"


def register_demo_ui(app: FastAPI) -> None:
    """Register the custom demo page and serve sibling CSS/JS assets."""
    if ASSETS_DIR.exists() and not any(
        getattr(route, "path", None) == "/demo/assets" for route in app.router.routes
    ):
        app.mount("/demo/assets", StaticFiles(directory=ASSETS_DIR), name="demo-assets")

    @app.get("/demo", include_in_schema=False)
    async def demo_redirect() -> RedirectResponse:
        # ensure trailing slash so relative asset paths resolve correctly
        return RedirectResponse(url="/demo/", status_code=308)

    @app.get("/demo/", include_in_schema=False)
    async def demo_page() -> FileResponse:
        return FileResponse(INDEX_HTML)

    # serve sibling files (styles.css, panels.css, map.css, demo.js, ...)
    # at /demo/<file>. Use a single catch-all so we don't have to enumerate.
    @app.get("/demo/{filename}", include_in_schema=False)
    async def demo_static(filename: str) -> FileResponse:
        from fastapi import HTTPException

        # restrict to the demo directory; reject path traversal
        safe = (DEMO_DIR / filename).resolve()
        if not str(safe).startswith(str(DEMO_DIR.resolve())) or not safe.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(safe)
