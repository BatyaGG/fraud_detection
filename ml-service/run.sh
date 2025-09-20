#!/usr/bin/env bash
set -e

export RUNNER="gunicorn"

gunicorn app:app \
         -b 0.0.0.0:8000 \
         --timeout=10 \
         --log-level=debug \
         --workers=8