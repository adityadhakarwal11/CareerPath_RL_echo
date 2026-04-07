from __future__ import annotations

from typing import Dict, List

import streamlit as st

from inference import choose_action
from openenv.env import CareerMobilityEnv
from openenv.models import Action, Observation
from openenv.tasks import TASKS, grade_task


ACTION_OPTIONS = [
    "upskill",
    "network",
    "tailor_resume",
    "apply_targeted",
    "apply_mass",
    "take_survival_job",
    "prepare_interview",
]


def ensure_session(task_id: str, college_tier: int) -> None:
    env: CareerMobilityEnv | None = st.session_state.get("env")
    env_task = st.session_state.get("task_id")
    env_tier = st.session_state.get("college_tier")

    if env is None or env_task != task_id or env_tier != college_tier:
        env = CareerMobilityEnv(task_id=task_id, college_tier=college_tier)
        st.session_state.env = env
        st.session_state.task_id = task_id
        st.session_state.college_tier = college_tier
        st.session_state.trace = [env.state().model_dump()]
        st.session_state.reward_trace = []
        st.session_state.action_trace = []


def reset_env() -> None:
    env: CareerMobilityEnv = st.session_state.env
    obs = env.reset()
    st.session_state.trace = [obs.model_dump()]
    st.session_state.reward_trace = []
    st.session_state.action_trace = []


def step_env(action_type: str) -> None:
    env: CareerMobilityEnv = st.session_state.env
    obs, reward, done, info = env.step(Action(action_type=action_type))
    st.session_state.trace.append(obs.model_dump())
    st.session_state.reward_trace.append(
        {
            "day_index": obs.day_index,
            "reward_total": reward.total,
            "progress": reward.progress,
            "quality": reward.quality,
            "penalty": reward.penalty,
            "bonus": reward.bonus,
            "done": done,
            "notes": " | ".join(info.notes),
        }
    )
    st.session_state.action_trace.append(
        {
            "day_index": obs.day_index,
            "action": action_type,
            "career_stage": obs.career_stage,
            "callback_probability": info.success_probability,
        }
    )


def autoplay(max_steps: int = 40) -> None:
    env: CareerMobilityEnv = st.session_state.env
    done = False
    while not done and env.state().day_index < max_steps:
        action_type = choose_action(env.state())
        step_env(action_type)
        done = bool(st.session_state.reward_trace[-1]["done"])


def render_metrics(obs: Observation, task_id: str) -> None:
    grade = grade_task(task_id, obs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Career Stage", obs.career_stage)
    col2.metric("Readiness", f"{obs.interview_readiness:.2f}")
    col3.metric("Callbacks", obs.callback_count)
    col4.metric("Task Score", f"{grade['score']:.2f}")


def main() -> None:
    st.set_page_config(page_title="Career Mobility RL Dashboard", layout="wide")
    st.title("Career Mobility RL Dashboard")
    st.caption("Inspect state transitions, step actions manually, and verify grader outputs for the RL environment.")

    with st.sidebar:
        st.header("Environment")
        task_id = st.selectbox(
            "Task",
            options=list(TASKS.keys()),
            format_func=lambda task: TASKS[task].title,
        )
        college_tier = st.slider("College Tier", min_value=1, max_value=5, value=5, step=1)
        ensure_session(task_id, college_tier)

        if st.button("Reset Environment", use_container_width=True):
            reset_env()
        if st.button("Run Baseline Policy", use_container_width=True):
            autoplay()

        manual_action = st.selectbox("Manual Action", options=ACTION_OPTIONS)
        if st.button("Step Action", use_container_width=True):
            step_env(manual_action)

    env: CareerMobilityEnv = st.session_state.env
    obs = env.state()
    render_metrics(obs, task_id)

    st.subheader("Current Observation")
    st.json(obs.model_dump())

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("State Trajectory")
        trace: List[Dict[str, object]] = st.session_state.trace
        if len(trace) > 1:
            chart_rows = [
                {
                    "day_index": row["day_index"],
                    "skill_score": row["skill_score"],
                    "resume_score": row["resume_score"],
                    "network_score": row["network_score"],
                    "interview_readiness": row["interview_readiness"],
                    "callback_probability": row["callback_probability"],
                }
                for row in trace
            ]
            st.line_chart(chart_rows, x="day_index")
        else:
            st.info("Take a step or run the baseline policy to populate the trajectory chart.")

        st.subheader("Reward Trace")
        if st.session_state.reward_trace:
            st.dataframe(st.session_state.reward_trace, use_container_width=True)
        else:
            st.info("No rewards yet.")

    with col_right:
        st.subheader("Action Log")
        if st.session_state.action_trace:
            st.dataframe(st.session_state.action_trace, use_container_width=True)
        else:
            st.info("No actions taken yet.")

        st.subheader("Task Details")
        task = TASKS[task_id]
        st.write(task.objective)
        st.json(task.success_metrics)

        st.subheader("Verifier Snapshot")
        st.json(grade_task(task_id, obs))


if __name__ == "__main__":
    main()
