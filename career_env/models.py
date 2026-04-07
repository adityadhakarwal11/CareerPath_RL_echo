from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


CareerStage = Literal["searching", "survival", "interview_ready", "placed", "growth"]
ActionType = Literal[
    "upskill",
    "network",
    "tailor_resume",
    "apply_targeted",
    "apply_mass",
    "take_survival_job",
    "prepare_interview",
]


class Observation(BaseModel):
    day_index: int = Field(ge=0)
    college_tier: int = Field(ge=1, le=5)
    skill_score: float = Field(ge=0.0, le=1.0)
    resume_score: float = Field(ge=0.0, le=1.0)
    network_score: float = Field(ge=0.0, le=1.0)
    savings_score: float = Field(ge=0.0, le=1.0)
    debt_score: float = Field(ge=0.0, le=1.0)
    experience_months: int = Field(ge=0)
    application_quality: float = Field(ge=0.0, le=1.0)
    interview_readiness: float = Field(ge=0.0, le=1.0)
    callback_probability: float = Field(ge=0.0, le=1.0)
    application_count: int = Field(ge=0)
    callback_count: int = Field(ge=0)
    low_quality_streak: int = Field(ge=0)
    career_stage: CareerStage


class Action(BaseModel):
    action_type: ActionType


class RewardBreakdown(BaseModel):
    total: float
    progress: float
    quality: float
    penalty: float
    bonus: float
    reasons: List[str]


class StepInfo(BaseModel):
    success_probability: float = Field(ge=0.0, le=1.0)
    stage_changed: bool
    task_id: str
    notes: List[str]


class TaskSpec(BaseModel):
    id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    objective: str
    max_days: int
    success_metrics: Dict[str, float]
