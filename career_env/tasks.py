from __future__ import annotations

from typing import Dict

from .models import Observation, TaskSpec


TASKS = {
    "easy-survival-stability": TaskSpec(
        id="easy-survival-stability",
        difficulty="easy",
        title="Easy: Survival To Interview Readiness",
        objective="Avoid collapse while building enough readiness to start targeted applications.",
        max_days=14,
        success_metrics={
            "interview_readiness": 0.65,
            "application_quality": 0.55,
            "debt_score_max": 0.92,
        },
    ),
    "medium-targeted-progress": TaskSpec(
        id="medium-targeted-progress",
        difficulty="medium",
        title="Medium: Targeted Progress",
        objective="Turn a disadvantaged starting point into a stable, high-quality application trajectory.",
        max_days=21,
        success_metrics={
            "callback_count": 1.0,
            "application_quality": 0.68,
            "career_stage_min": 2.0,
        },
    ),
    "hard-upward-mobility": TaskSpec(
        id="hard-upward-mobility",
        difficulty="hard",
        title="Hard: Upward Mobility",
        objective="Reach placement or growth-stage through realistic sequencing of skill-building, networking, and job search.",
        max_days=30,
        success_metrics={
            "callback_count": 2.0,
            "interview_readiness": 0.8,
            "career_stage_min": 3.0,
            "application_quality": 0.74,
        },
    ),
}


STAGE_TO_SCORE = {
    "searching": 0.0,
    "survival": 1.0,
    "interview_ready": 2.0,
    "placed": 3.0,
    "growth": 4.0,
}


def grade_task(task_id: str, obs: Observation) -> Dict[str, float]:
    task = TASKS[task_id]
    metrics = []

    if "interview_readiness" in task.success_metrics:
        metrics.append(min(obs.interview_readiness / task.success_metrics["interview_readiness"], 1.0))
    if "application_quality" in task.success_metrics:
        metrics.append(min(obs.application_quality / task.success_metrics["application_quality"], 1.0))
    if "callback_count" in task.success_metrics:
        metrics.append(min(obs.callback_count / task.success_metrics["callback_count"], 1.0))
    if "career_stage_min" in task.success_metrics:
        metrics.append(min(STAGE_TO_SCORE[obs.career_stage] / task.success_metrics["career_stage_min"], 1.0))
    if "debt_score_max" in task.success_metrics:
        debt_component = (
            1.0
            if obs.debt_score <= task.success_metrics["debt_score_max"]
            else max(0.0, 1.0 - (obs.debt_score - task.success_metrics["debt_score_max"]))
        )
        metrics.append(debt_component)

    raw_score = sum(metrics) / max(len(metrics), 1)
    score = min(max(raw_score, 0.001), 0.999)

    return {
        "score": round(score, 4),
        "metrics_used": float(len(metrics)),
    }
