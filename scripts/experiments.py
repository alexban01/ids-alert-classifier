#!/usr/bin/env python
"""
experiments.py — Rebuild EXPERIMENTS.md from every models/*/run.json.

Run from the project root:
    .venv/bin/python scripts/experiments.py

The sidecar run.json files are written automatically by train.py (settings) and
benchmark_realworld.py (results); this just regenerates the human-readable ledger
on demand (e.g. after pulling adapters back from RunPod, or editing a run.json).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ids.run_manifest import EXPERIMENTS_MD, regenerate_experiments_md

if __name__ == "__main__":
    n = regenerate_experiments_md()
    print(f"✅ {EXPERIMENTS_MD} rebuilt from {n} run(s).")
