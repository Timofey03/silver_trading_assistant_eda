#!/usr/bin/env bash
# .devcontainer/start.sh — стартует backend + frontend на каждом старте Codespace
set -e

cd /workspaces/$(basename "$(pwd)" 2>/dev/null || basename "$PWD")

echo "=== Starting Argentum services ==="

# Backend в фоне
echo "[Backend] uvicorn :8000"
cd argentum/backend
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 \
  > /tmp/argentum-backend.log 2>&1 &
BACKEND_PID=$!
cd ../..
echo "  PID $BACKEND_PID, logs: /tmp/argentum-backend.log"

# Wait for backend ready
for i in {1..20}; do
  if curl -s -o /dev/null http://127.0.0.1:8000/api/health; then
    echo "  Backend ready ✓"
    break
  fi
  sleep 1
done

# Frontend в фоне
echo "[Frontend] next dev :3000"
cd argentum/frontend
nohup npm run dev > /tmp/argentum-frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..
echo "  PID $FRONTEND_PID, logs: /tmp/argentum-frontend.log"

echo ""
echo "=== Services starting ==="
echo "Откройте порт 3000 для UI (forwarded автоматически)."
echo "Backend на 8000 (только internal use)."
echo ""
echo "tail -f /tmp/argentum-{backend,frontend}.log для логов"
