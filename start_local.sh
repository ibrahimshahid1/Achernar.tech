#!/bin/bash

# Trap Ctrl+C to kill all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Source 1: Django Backend
echo "Starting Django Backend..."
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend
python manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!

# Source 2: Rust Daemon
echo "Starting Rust Daemon..."
cd achernar-daemon
# Ensure cargo env is sourced if needed, though usually available in shell
# source $HOME/.cargo/env 
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
