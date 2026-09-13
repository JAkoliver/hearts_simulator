@echo off
REM League r9 PAUSE (docs/exploiter_league_r9_prereg.md §9): kill every research
REM process tree - trainer, probes, gates, SearchEval - so NOTHING holds VRAM or
REM CPU while paused. Never suspends. Resume = re-run the stage driver; every
REM stage skips completed units (trainer: --resume from train_ckpt.pt).
echo Pausing league r9 work (kill, not suspend)...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -match 'python|SearchEval|bash') -and ($_.CommandLine -match 'train.py|defense_probe|run_match_gate|neutral_raw_eval|SearchEval|run_r9_|run_r8_|validate_') } | ForEach-Object { Write-Host ('  kill ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 3 /nobreak > nul
powershell -NoProfile -Command "$n = (Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|SearchEval' -and $_.CommandLine -match 'train.py|SearchEval|defense_probe|run_match_gate' } | Measure-Object).Count; Write-Host ('remaining research processes: ' + $n)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo Paused. Resume with the stage driver (trainer: python train.py --resume).
