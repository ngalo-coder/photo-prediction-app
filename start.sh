#!/usr/bin/env bash
poetry run gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --timeout 120