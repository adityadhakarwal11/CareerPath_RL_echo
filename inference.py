from __future__ import annotations

import json
import os
from typing import Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

from career_env.env import CareerMobilityEnv
from career_env.models import Action, Observation
from career_env.tasks import TASKS, grade_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

VALID_ACTIONS = {
    "upskill",
    "network",
    "tailor_resume",
    "apply_targeted",
    "apply_mass",
    "take_survival_job",
    "prepare_interview",
}


def build_client() -> OpenAI | None:
    if OpenAI is None:
        return None
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def log_event(event: str, **fields: object) -> None:
    print(f"[{event}] {json.dumps(fields, ensure_ascii=True, sort_keys=True)}")


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


def choose_action_with_llm(obs: Observation, client: OpenAI) -> str:
    prompt = (
        "You are selecting one action for a career mobility RL environment. "
        "Return exactly one action token from this list: "
        "upskill, network, tailor_resume, apply_targeted, apply_mass, "
        "take_survival_job, prepare_interview.\n"
        f"State: {obs.model_dump_json()}"
    )
    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        max_output_tokens=16,
    )
    action_text = (response.output_text or "").strip()
    if action_text in VALID_ACTIONS:
        return action_text
    return choose_action(obs)


def run_task(task_id: str) -> Dict[str, float]:
    env = CareerMobilityEnv(task_id=task_id, college_tier=5)
    client = build_client()
    obs = env.reset()
    log_event(
        "START",
        task_id=task_id,
        task_title=TASKS[task_id].title,
        model_name=MODEL_NAME,
        api_base_url=API_BASE_URL,
        llm_enabled=bool(client),
        local_image_name=LOCAL_IMAGE_NAME,
    )

    done = False
    while not done:
        action_type = choose_action_with_llm(obs, client) if client else choose_action(obs)
        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)
        log_event(
            "STEP",
            task_id=task_id,
            day=obs.day_index,
            action=action_type,
            reward=reward.total,
            stage=obs.career_stage,
            callback_probability=info.success_probability,
        )

    result = grade_task(task_id, obs)
    log_event("END", task_id=task_id, score=result["score"], metrics_used=result["metrics_used"])
    return result


def main() -> None:
    scores: List[float] = []
    for task_id in TASKS:
        result = run_task(task_id)
        scores.append(result["score"])

    log_event("END", task_id="summary", score=round(sum(scores) / len(scores), 4), metrics_used=len(scores))


if __name__ == "__main__":
    main()
