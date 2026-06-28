"""Static and FastAPI dashboard entry points."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from robot_sorting.dashboard.dashboard_schemas import DashboardData
from robot_sorting.dashboard.run_data_loader import build_dashboard_data


def write_static_dashboard(
    output_dir: Path,
    dashboard_data: DashboardData | None = None,
    dashboard_output: Path | None = None,
) -> Path:
    """Write static dashboard files and return the dashboard directory."""

    data = dashboard_data or build_dashboard_data(output_dir)
    dashboard_dir = dashboard_output or output_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    serialized = data.model_dump(mode="json")
    (output_dir / "dashboard_data.json").write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (dashboard_dir / "dashboard_data.json").write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    static_dir = Path(__file__).parent / "static"
    app_js = static_dir / "app.js"
    styles_css = static_dir / "styles.css"
    shutil.copyfile(app_js, dashboard_dir / "app.js")
    shutil.copyfile(styles_css, dashboard_dir / "styles.css")
    template = (static_dir / "index.html").read_text(encoding="utf-8")
    embedded_data = json.dumps(serialized, ensure_ascii=False)
    html = template.replace("__DASHBOARD_DATA__", embedded_data)
    (dashboard_dir / "index.html").write_text(html, encoding="utf-8")
    return dashboard_dir


def create_dashboard_api(output_dir: Path) -> FastAPI:
    """Create a small FastAPI dashboard server for one run directory."""

    dashboard_dir = output_dir / "dashboard"
    if not (dashboard_dir / "index.html").exists():
        write_static_dashboard(output_dir)
    app = FastAPI(title="Robot Sorting Automation Cell Dashboard")
    app.mount("/static", StaticFiles(directory=dashboard_dir), name="dashboard-static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return (dashboard_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/api/runs/latest")
    def latest_run() -> dict[str, object]:
        return build_dashboard_data(output_dir).model_dump(mode="json")

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict[str, object]:
        _validate_run_id(output_dir, run_id)
        return build_dashboard_data(output_dir).summary.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/timeline")
    def run_timeline(run_id: str) -> list[dict[str, object]]:
        _validate_run_id(output_dir, run_id)
        return build_dashboard_data(output_dir).timeline

    @app.get("/api/runs/{run_id}/objects")
    def run_objects(run_id: str) -> list[dict[str, object]]:
        _validate_run_id(output_dir, run_id)
        return [row.model_dump(mode="json") for row in build_dashboard_data(output_dir).objects]

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> list[dict[str, object]]:
        _validate_run_id(output_dir, run_id)
        return build_dashboard_data(output_dir).events

    return app


def _validate_run_id(output_dir: Path, run_id: str) -> None:
    if run_id not in {output_dir.name, "latest", "run"}:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
