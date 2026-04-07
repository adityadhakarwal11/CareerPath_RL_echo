from __future__ import annotations

from typing import Dict, Tuple

from .models import Action, Observation, RewardBreakdown, StepInfo
from .tasks import TASKS


class CareerMobilityEnv:
    def __init__(self, task_id: str = "medium-targeted-progress", college_tier: int = 5):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.college_tier = college_tier
        self._obs = self._initial_observation(college_tier)

    def _initial_observation(self, college_tier: int) -> Observation:
        tier_penalty = (college_tier - 1) * 0.05
        return Observation(
            day_index=0,
            college_tier=college_tier,
            skill_score=max(0.18, 0.34 - tier_penalty),
            resume_score=max(0.2, 0.38 - tier_penalty),
            network_score=max(0.12, 0.26 - tier_penalty),
            savings_score=max(0.2, 0.44 - tier_penalty),
            debt_score=min(0.95, 0.56 + tier_penalty),
            experience_months=0,
            application_quality=0.22,
            interview_readiness=0.2,
            callback_probability=0.08,
            application_count=0,
            callback_count=0,
            low_quality_streak=0,
            career_stage="searching",
        )

    def reset(self) -> Observation:
        self._obs = self._initial_observation(self.college_tier)
        return self._obs

    def state(self) -> Observation:
        return self._obs

    def _update_stage(self, obs: Observation) -> str:
        if obs.callback_count >= 2 and obs.interview_readiness >= 0.82:
            return "growth"
        if obs.callback_count >= 1 and obs.interview_readiness >= 0.75:
            return "placed"
        if obs.interview_readiness >= 0.62:
            return "interview_ready"
        if obs.experience_months >= 2:
            return "survival"
        return "searching"

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def step(self, action: Action) -> Tuple[Observation, RewardBreakdown, bool, StepInfo]:
        obs = self._obs.model_copy(deep=True)
        obs.day_index += 1

        progress = 0.0
        quality = 0.0
        penalty = 0.0
        bonus = 0.0
        notes = []

        if action.action_type == "upskill":
            obs.skill_score = self._clamp(obs.skill_score + 0.12)
            obs.interview_readiness = self._clamp(obs.interview_readiness + 0.05)
            progress += 0.12
            notes.append("Skill score improved.")
        elif action.action_type == "network":
            obs.network_score = self._clamp(obs.network_score + 0.1)
            obs.callback_probability = self._clamp(obs.callback_probability + 0.05)
            progress += 0.08
            notes.append("Network strength improved.")
        elif action.action_type == "tailor_resume":
            obs.resume_score = self._clamp(obs.resume_score + 0.13)
            obs.application_quality = self._clamp(obs.application_quality + 0.12)
            obs.interview_readiness = self._clamp(obs.interview_readiness + 0.06)
            quality += 0.14
            notes.append("Resume quality improved.")
        elif action.action_type == "apply_targeted":
            obs.application_count += 1
            success_probability = self._clamp(
                0.15
                + 0.35 * obs.skill_score
                + 0.25 * obs.resume_score
                + 0.15 * obs.network_score
                + 0.2 * obs.application_quality
                - 0.03 * (obs.college_tier - 1)
            )
            obs.callback_probability = self._clamp(success_probability)
            if success_probability >= 0.56:
                obs.callback_count += 1
                bonus += 0.35
                obs.low_quality_streak = 0
                notes.append("Targeted application produced a callback.")
            else:
                obs.low_quality_streak += 1
                quality += 0.1
                notes.append("Targeted application improved trajectory but did not convert yet.")
        elif action.action_type == "apply_mass":
            obs.application_count += 2
            obs.application_quality = self._clamp(obs.application_quality - 0.08)
            obs.callback_probability = self._clamp(obs.callback_probability - 0.04)
            obs.low_quality_streak += 1
            penalty += 0.14
            notes.append("Mass apply hurt application quality.")
        elif action.action_type == "take_survival_job":
            obs.experience_months += 1
            obs.savings_score = self._clamp(obs.savings_score + 0.08)
            obs.debt_score = self._clamp(obs.debt_score - 0.04)
            progress += 0.08
            notes.append("Short-term stability improved.")
        elif action.action_type == "prepare_interview":
            obs.interview_readiness = self._clamp(obs.interview_readiness + 0.14)
            obs.callback_probability = self._clamp(obs.callback_probability + 0.06)
            quality += 0.12
            notes.append("Interview readiness improved.")

        if obs.low_quality_streak >= 2:
            penalty += 0.08
            notes.append("Repeated low-value behavior triggered a streak penalty.")

        new_stage = self._update_stage(obs)
        stage_changed = new_stage != obs.career_stage
        obs.career_stage = new_stage

        if stage_changed:
            bonus += 0.2
            notes.append(f"Career stage advanced to {new_stage}.")

        total = round(progress + quality + bonus - penalty, 4)
        reward = RewardBreakdown(
            total=total,
            progress=round(progress, 4),
            quality=round(quality, 4),
            penalty=round(-penalty, 4),
            bonus=round(bonus, 4),
            reasons=notes,
        )

        done = obs.day_index >= self.task.max_days or obs.career_stage in {"placed", "growth"}
        info = StepInfo(
            success_probability=obs.callback_probability,
            stage_changed=stage_changed,
            task_id=self.task_id,
            notes=notes,
        )
        self._obs = obs
        return obs, reward, done, info
