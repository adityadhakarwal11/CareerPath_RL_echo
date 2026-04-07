from __future__ import annotations

import json
import os
from typing import Dict, List

from openenv.env import CareerMobilityEnv
from openenv.models import Action, Observation
from openenv.tasks import TASKS, grade_task


def choose_action(obs: Observation) -> str:
    if obs.skill_score < 0.55:
        return "upskill"
    if obs.resume_score < 0.62 or obs.application_quality < 0.58:
        return "tailor_resume"
    if obs.network_score < 0.5:
        return "network"
    if obs.interview_readiness < 0.72 and obs.callback_count == 0:
        return "prepare_interview"
    if obs.application_quality >= 0.6:
        return "apply_targeted"
    return "take_survival_job"


def run_task(task_id: str) -> Dict[str, float]:
    env = CareerMobilityEnv(task_id=task_id, college_tier=5)
    obs = env.reset()
    print(json.dumps({"type": "START", "task_id": task_id, "task_title": TASKS[task_id].title}))

    done = False
    while not done:
        action_type = choose_action(obs)
        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)
        print(
            json.dumps(
                {
                    "type": "STEP",
                    "task_id": task_id,
                    "day": obs.day_index,
                    "action": action_type,
                    "reward": reward.total,
                    "stage": obs.career_stage,
                    "callback_probability": info.success_probability,
                }
            )
        )

    result = grade_task(task_id, obs)
    print(json.dumps({"type": "END", "task_id": task_id, "score": result["score"]}))
    return result


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "")
    hf_token = os.getenv("HF_TOKEN", "")
    print(
        json.dumps(
            {
                "type": "CONFIG",
                "api_base_url_present": bool(api_base_url),
                "model_name_present": bool(model_name),
                "hf_token_present": bool(hf_token),
            }
        )
    )

    scores: List[float] = []
    for task_id in TASKS:
        result = run_task(task_id)
        scores.append(result["score"])

    print(json.dumps({"type": "SUMMARY", "mean_score": round(sum(scores) / len(scores), 4)}))


if __name__ == "__main__":
    main()
