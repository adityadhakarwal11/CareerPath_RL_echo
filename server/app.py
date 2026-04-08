from __future__ import annotations

from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from career_env.env import CareerMobilityEnv
from career_env.models import Action, Observation, RewardBreakdown, StepInfo
from career_env.tasks import TASKS


ROOT = Path(__file__).resolve().parent.parent
OPENENV_PATH = ROOT / "openenv.yaml"
CURRENT_ENV = CareerMobilityEnv()

app = FastAPI(title="Career Mobility OpenEnv", version="0.1.0")


class ResetRequest(BaseModel):
    task_id: str = Field(default="medium-targeted-progress")
    college_tier: int = Field(default=5, ge=1, le=5)


class StepResponse(BaseModel):
    observation: Observation
    reward: RewardBreakdown
    done: bool
    info: StepInfo


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    task_links = "".join(f"<li><strong>{task_id}</strong>: {task.title}</li>" for task_id, task in TASKS.items())
    return (
        "<html><body style='font-family:Arial,sans-serif;max-width:920px;margin:40px auto;padding:0 20px;'>"
        "<h1>Career Mobility OpenEnv</h1>"
        "<p>This service exposes the validator-facing OpenEnv endpoints for the career mobility RL environment.</p>"
        "<ul>"
        "<li>POST /reset</li>"
        "<li>POST /step</li>"
        "<li>GET /state</li>"
        "<li>GET /openenv.yaml</li>"
        "<li>GET /tasks</li>"
        "</ul>"
        "<h2>Tasks</h2><ul>"
        f"{task_links}"
        "</ul>"
        "</body></html>"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, object]:
    return {
        "tasks": [
            {
                "id": task.id,
                "difficulty": task.difficulty,
                "title": task.title,
                "objective": task.objective,
                "max_days": task.max_days,
                "success_metrics": task.success_metrics,
            }
            for task in TASKS.values()
        ]
    }


@app.get("/openenv.yaml")
def openenv_yaml() -> FileResponse:
    return FileResponse(OPENENV_PATH, media_type="text/yaml", filename="openenv.yaml")


@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = None) -> Observation:
    global CURRENT_ENV
    payload = request or ResetRequest()
    CURRENT_ENV = CareerMobilityEnv(task_id=payload.task_id, college_tier=payload.college_tier)
    return CURRENT_ENV.reset()


@app.get("/state", response_model=Observation)
def state() -> Observation:
    return CURRENT_ENV.state()


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    observation, reward, done, info = CURRENT_ENV.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
