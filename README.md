# Career Mobility RL Submission

This repository contains a reinforcement-learning-first career mobility environment, a deterministic baseline inference script, and a Streamlit dashboard for inspecting environment state transitions and grader outputs.

## Included Files

- `openenv/` : typed models, environment logic, task specs, graders, and `openenv.yaml`
- `inference.py` : reproducible baseline policy runner
- `streamlit_app.py` : interactive dashboard to inspect the environment
- `requirements-openenv.txt` : lightweight Python dependencies for the RL side

## How To Run

```powershell
pip install -r requirements-openenv.txt
python inference.py
streamlit run streamlit_app.py
```

## What This Project Simulates

A disadvantaged job seeker improves their long-term career trajectory through actions such as:

- upskill
- network
- tailor_resume
- apply_targeted
- apply_mass
- take_survival_job
- prepare_interview

The environment rewards meaningful upward progress and penalizes low-quality or spam-like behavior.

## Tasks

- `easy-survival-stability`
- `medium-targeted-progress`
- `hard-upward-mobility`

See `openenv/README.md` for the detailed environment design.
