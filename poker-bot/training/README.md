# v.5 training — start here

The root `agent.py` and new `agent_v4.py` are frozen v.4 baselines. `agent_v5.py` is the policy candidate. `judge_engine.py`, `game.py`, and `evaluator.py` are unchanged.

All commands from the **poker-bot repository root**. Use Python 3.10+ with `fastapi`, `uvicorn`, and `pytest` installed; don't reuse the bundled macOS `.venv` on another machine.

```bash
python3 -m pip install -r requirements.txt -r requirements-dev.txt
python3 -m py_compile agent_v5.py training/train.py training/export.py
```

## Quick smoke check (few matches)

`train.py` launches and shuts down its own servers on ports 5610 and 5611. Ensure those ports aren't already occupied. Unlike a manual `judge_engine.py` test, you do not launch agents first.

```bash
python3 training/train.py --iterations 0 --deals 100 --seeds 1,2 --opponents v3c,v4,random
```

This evaluates default v.5 parameters and leaves detailed judge JSON under `training/results/run_*/`.

## Optimize across more than one opponent

```bash
python3 training/train.py --iterations 12 --deals 300 --seeds 10,11,12 --opponents v3c,v4,v3,random,calling,half,pot
```

The population uses reproducible random local mutations of ten thresholds, selects on average return plus **worst-opponent** performance, and saves `training/models/best.json`. This is a cheap black-box search, **not** a guaranteed optimal or proven winning strategy. Repeat optimization with different training seeds only when appropriate; never treat those seeds as held-out tests.

## Held-out evaluation, plus preserved v.4 baseline

Use different seeds from optimization. Evaluate both on identical seeds and opponents, each using a complete paired-deal judge match. Only compare the corresponding JSON summaries, not one lucky seed.

```bash
python3 training/train.py --iterations 0 --deals 1000 --seeds 201,202,203 --opponents v3c,v4,v3,random,calling,half,pot --start training/models/best.json
python3 training/train.py --iterations 0 --deals 1000 --seeds 201,202,203 --opponents v3c,v3,random,calling,half,pot --bot v4
```

The v.4 baseline should not be compared against itself, so it omits the `v4` opponent. Note that each v.5-v4 matchup compares v.5 directly to v.4; the v.4 baseline evaluation only compares v.4 to others.

## Export for tournament

```bash
python3 training/export.py --model training/models/best.json --output agent_v5_submission.py
cp agent.py agent_v4_backup.py
cp agent_v5_submission.py agent.py
python3 -m pytest -q
```

You may also run `./tools/check_build.sh` after making the replacement. Check any resulting model or code before submission. Never submit unvalidated output; target an improvement against v3c **and** no catastrophic regression elsewhere.

## How live learning works

v.5 counts each observed opponent action once. It tracks half/pot raises per street, whether the opponent attacked immediately after our checks, and calls/folds after our raises. Smoothed estimates and experience gating inform a small amount of additional bluffing/defense. All state is in tournament `memory`; no private opponent cards are assumed during play, and no state carries between halves.

The original source policy remains runnable with `PORT=5610 python3 agent_v5.py`; for a locally optimized candidate, the training harness passes parameters in `P5_PARAMS_JSON`. The exported file bakes the chosen parameters into defaults, so the tournament needs no environment variable or external file.
