@echo off
cd /d E:\hearts_simulator
echo Starting Hearts vs AI web server...
echo Open http://localhost:8642 in your browser once it says "Uvicorn running".
python -m uvicorn hearts_web.server:app --host 127.0.0.1 --port 8642
pause
