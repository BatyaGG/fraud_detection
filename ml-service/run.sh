#!/usr/bin/env bash
set -e
export FLASK_APP=app.py
export APP_VERSION="lgbm_iforest_2025-09-16"
python app.py
