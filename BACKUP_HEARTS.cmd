@echo off
REM Hearts project backup to H: (docs/RELEASE_PLAN.md sec. 2/5).
REM Archive-safe: /E copies new+changed, never deletes on the target -
REM a source-side loss cannot propagate into the backup.
set DEST=H:\hearts_backup
set SRC=E:\hearts_simulator
if not exist %DEST% mkdir %DEST%
echo ===== backup start %DATE% %TIME% ===== >> %DEST%\backup.log

REM Full git history as a single restorable file (2-generation rotation).
if exist %DEST%\repo_latest.bundle copy /y %DEST%\repo_latest.bundle %DEST%\repo_prev.bundle >nul
"C:\Program Files\Git\bin\git.exe" -C %SRC% bundle create %DEST%\repo_latest.bundle --all >> %DEST%\backup.log 2>&1

REM Model milestones, anchors, root checkpoints/traces.
robocopy %SRC%\Hall_of_Fame %DEST%\Hall_of_Fame /E /Z /R:2 /W:5 /NP /NDL >> %DEST%\backup.log
robocopy %SRC%\legacy_v3_pass238 %DEST%\legacy_v3_pass238 /E /Z /R:2 /W:5 /NP /NDL >> %DEST%\backup.log
robocopy %SRC% %DEST%\root_models *.pth *.pt /Z /R:2 /W:5 /NP /NDL >> %DEST%\backup.log

REM Datasets: validation + verdicts + expert-iteration banks.
robocopy %SRC%\equity_data %DEST%\equity_data /E /Z /R:2 /W:5 /NP /NDL >> %DEST%\backup.log
robocopy %SRC%\expert_data %DEST%\expert_data /E /Z /R:2 /W:5 /NP /NDL >> %DEST%\backup.log

REM Web app (demo path) + docs are inside the repo bundle already, but the
REM match logs are gitignored personal data - keep a private copy.
robocopy %SRC%\hearts_web %DEST%\hearts_web /E /Z /R:2 /W:5 /NP /NDL >> %DEST%\backup.log

echo ===== backup done %DATE% %TIME% ===== >> %DEST%\backup.log
