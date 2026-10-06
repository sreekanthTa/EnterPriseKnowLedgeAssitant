from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse


BASE_DIR = Path(__file__).resolve().parent
RETRIEVAL_PATH = BASE_DIR / "02.retrieval.py"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("retrieval", RETRIEVAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load retrieval module from {RETRIEVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = FastAPI(title="Enterprise Knowledge Assistant Health API")


@app.get("/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    try:
        health = load_pipeline_module().health_check()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy"},
        )

    status_code = 200 if health.get("ok") else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if status_code == 200 else "unhealthy",
            "checks": health,
        },
    )
