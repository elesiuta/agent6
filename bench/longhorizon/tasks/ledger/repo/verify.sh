#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 tools/gen_cli.py
exec python3 -m unittest discover -s tests -t . -p 'test_*.py'
