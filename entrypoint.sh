#!/usr/bin/env bash
set -e

# If the data folder is missing or empty, run dataset setup
if [ ! -d "./dataset" ] || [ -z "$(ls -A ./dataset)" ]; then
  echo ">>> Running initial dataset setup..."
  python dataset_setup.py
  echo ">>> Dataset setup complete."
fi

exec "$@"
