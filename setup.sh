#!/usr/bin/env bash
# One-time setup: create venv and install dependencies
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Then run: python main.py --help"
