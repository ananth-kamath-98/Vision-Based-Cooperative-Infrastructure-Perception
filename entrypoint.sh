#!/usr/bin/env bash
set -e

# If the data folder is missing or empty, run dataset setup
if [ ! -d "./dataset/video_clips" ] || [ -z "$(ls -A ./dataset/video_clips)" ]; then
  echo ">>> Running initial dataset setup..."
  python ./dataset_setup/dataset_setup.py
  echo ">>> Dataset setup complete."
fi

exec "$@"
