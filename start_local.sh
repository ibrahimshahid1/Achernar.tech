#!/bin/bash

# Trap Ctrl+C to kill all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Source 1: Django Backend
echo "Starting Django Backend..."
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend
export ACHERNAR_RELAY_DEV_TOKEN=${ACHERNAR_RELAY_DEV_TOKEN:-achernar-dev-token}
python manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!

# Source 2: Rust Daemon
echo "Starting Rust Daemon..."
cd achernar-daemon
# Ensure cargo env is sourced if needed, though usually available in shell
# source $HOME/.cargo/env 
export ACHERNAR_HOST=${ACHERNAR_HOST:-127.0.0.1}
export ACHERNAR_PORT=${ACHERNAR_PORT:-3030}
export ACHERNAR_CLOUD_HTTP=${ACHERNAR_CLOUD_HTTP:-http://127.0.0.1:8000}
export ACHERNAR_CLOUD_WS=${ACHERNAR_CLOUD_WS:-ws://127.0.0.1:8000}
export ACHERNAR_HARDWARE_ID=${ACHERNAR_HARDWARE_ID:-achernar-local-daemon}
cargo run &
DAEMON_PID=$!
cd ..

# Source 3: React Frontend
echo "Starting React Frontend..."
cd achernar-frontend
pnpm dev &
FRONTEND_PID=$!
cd ..

echo "All services started!"
echo "Django: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Daemon: http://localhost:3030"
echo "Press Ctrl+C to stop all services."

wait
