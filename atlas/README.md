# Atlas App

## Purpose

Stores celestial catalogs and astronomical object data (Messier, NGC, IC, Gaia).

## Models

- Celestial object catalog entries
- Coordinates and metadata
- Observation history

## API Endpoints

- `GET /api/atlas/objects/` — Search celestial objects
- `GET /api/atlas/objects/<id>/` — Get object details

## Dependencies

- Used by: `planning` app for target scheduling

## PDF Reference

Section 16.3 - Django Backend Services → Target Database
