# Career Mobility RL OpenEnv Starter

This folder now contains a more RL-native version of the earlier idea: a career mobility environment where a disadvantaged job seeker learns long-horizon strategies through skill-building, networking, resume improvement, targeted applications, and survival work.

## Why This Version

This design stays closer to classic reinforcement learning than a pure job recommender:

- sequential actions with delayed effects
- explicit environment state transitions
- meaningful reward shaping over full trajectories
- realistic constraints around employability and mobility
- multiple tasks with deterministic graders

## Environment

`CareerMobilityEnv` simulates upward mobility over a 14-30 day horizon depending on task difficulty.

The environment is intentionally real-world flavored:

- students from lower college tiers start with weaker resume, skill, and network signals
- targeted effort can improve long-term outcomes
- low-quality application spam is penalized
- survival work can stabilize short-term state while slowing growth

## Observation Model

Defined in [models.py](/C:/app%20work/job-match-app/openenv/models.py):

- `day_index`
- `college_tier`
- `skill_score`
- `resume_score`
- `network_score`
- `savings_score`
- `debt_score`
- `experience_months`
- `application_quality`
- `interview_readiness`
- `callback_probability`
- `application_count`
- `callback_count`
- `low_quality_streak`
- `career_stage`

## Actions

- `upskill`
- `network`
- `tailor_resume`
- `apply_targeted`
- `apply_mass`
- `take_survival_job`
- `prepare_interview`

## Career Stages

- `searching`
- `survival`
- `interview_ready`
- `placed`
- `growth`

## Reward Design

The reward is intentionally decomposed into:

- `progress`
- `quality`
- `penalty`
- `bonus`

Positive reward:

- improving skill score
- improving resume quality
- improving network strength
- increasing interview readiness
- earning callbacks
- advancing career stage

Negative reward:

- mass low-quality applications
- repeated low-value behavior
- degrading application quality

## Tasks

Defined in [tasks.py](/C:/app%20work/job-match-app/openenv/tasks.py):

1. `easy-survival-stability`
2. `medium-targeted-progress`
3. `hard-upward-mobility`

Each task has:

- a concrete objective
- a fixed horizon
- deterministic grading logic
- scores in the `0.0-1.0` range

## Files

- [models.py](/C:/app%20work/job-match-app/openenv/models.py): typed Observation, Action, Reward models
- [env.py](/C:/app%20work/job-match-app/openenv/env.py): environment logic
- [tasks.py](/C:/app%20work/job-match-app/openenv/tasks.py): task specs and graders
- [openenv.yaml](/C:/app%20work/job-match-app/openenv/openenv.yaml): environment metadata
- [inference.py](/C:/app%20work/job-match-app/inference.py): deterministic baseline inference

## Baseline Inference

The baseline policy is intentionally simple and reproducible:

- build skill first
- improve resume second
- strengthen network third
- prepare interview readiness
- use targeted applications once the state is strong enough

This keeps the environment RL-friendly while satisfying the hack requirement for a reproducible baseline run.
