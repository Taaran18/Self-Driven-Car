#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

[ -f "$BACKEND/.env" ] || cp "$BACKEND/.env.example" "$BACKEND/.env"
[ -f "$FRONTEND/.env" ] || cp "$FRONTEND/.env.example" "$FRONTEND/.env"

BACKEND_CMD="cd '$BACKEND' && { [ -d .venv ] || python3 -m venv .venv; } && source .venv/bin/activate && pip install -q -r requirements.txt && uvicorn app.main:app --reload --reload-dir app --port 8000"
FRONTEND_CMD="cd '$FRONTEND' && { [ -d node_modules ] || npm install; } && npm run dev"

osascript <<OSA
tell application "Terminal"
  activate
  do script "$BACKEND_CMD"
  do script "$FRONTEND_CMD"
end tell
OSA

echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
