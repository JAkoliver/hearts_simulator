@echo off
REM Resume round-2 defender-corpus generation from where it was paused.
bash -c "nohup bash ops/run_r2_gen.sh > logs/r2_nohup.log 2>&1 &"
echo generation resumed (logs/r2_gen.log for milestones)
pause
