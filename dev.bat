@echo off
setlocal
set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend"

if not exist "%BACKEND%\.env" copy "%BACKEND%\.env.example" "%BACKEND%\.env" >nul
if not exist "%FRONTEND%\.env" copy "%FRONTEND%\.env.example" "%FRONTEND%\.env" >nul

start "Self-Driven Car Backend" cmd /k "cd /d "%BACKEND%" && (if not exist .venv python -m venv .venv) && call .venv\Scripts\activate && pip install -q -r requirements.txt && uvicorn app.main:app --reload --reload-dir app --port 8000"
start "Self-Driven Car Frontend" cmd /k "cd /d "%FRONTEND%" && (if not exist node_modules npm install) && npm run dev"

echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
endlocal
