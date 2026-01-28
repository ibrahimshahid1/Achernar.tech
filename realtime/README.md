# Realtime App

## Purpose

Handles WebSocket connections and IoT relay for real-time communication between Rust daemons and web clients.

## Models

- **DaemonConnection**: Tracks active daemon connections (persistent WebSocket tunnels)
- **UserSession**: Tracks web client sessions subscribed to telemetry
- **TelemetrySnapshot**: Digital twin - latest telemetry state for each daemon

## WebSocket Endpoints

- `ws://localhost:8000/ws/gateway/<daemon_id>/` — Daemon persistent connection
- `ws://localhost:8000/ws/telemetry/<daemon_id>/` — Web client telemetry subscription

## Components

- `consumers.py` — WebSocket consumers (DaemonConsumer, TelemetryConsumer)
- `routing.py` — WebSocket URL routing
- `relay.py` — IoT relay helper functions
- `models.py` — Database models for connection tracking

## PDF Reference

Section 16.2 - Cloud-Mediated Remote Operations
