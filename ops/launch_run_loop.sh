#!/bin/bash
# Launcher for the PPO experimentation loop (run_loop.py).
# PYTHONUNBUFFERED is exported so the child interpreters run_loop spawns
# (orchestrator.py -> train.py / export.py) are unbuffered too; -u alone
# would only cover run_loop itself.
cd /e/hearts_simulator || exit 1
export PYTHONUNBUFFERED=1
export HEARTS_HEADROOM=0.25
exec python -u run_loop.py >> /e/hearts_simulator/logs/run_loop_20260728_match.log 2>&1
