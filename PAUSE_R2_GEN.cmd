@echo off
REM Lossless pause for round-2 defender-corpus generation (kills by PID
REM file; every completed match is already on disk, the one partial
REM match is trimmed automatically on resume).
bash ops/pause_r2_gen.sh
pause
