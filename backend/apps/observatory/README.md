# Observatory App

## Purpose

Manages single-user equipment profiles and observatory configurations.

## Models

- Equipment profiles
- Site locations
- Observatory settings

## API Endpoints

- `GET /api/observatory/profiles/` — List equipment profiles
- `POST /api/observatory/profiles/` — Create new profile

## Dependencies

- Uses: `equipment` app for hardware specs
- Used by: `imaging` app for session context

## PDF Reference

Section 16.2 - Cloud-Mediated Remote Operations (implied for observatory management)
