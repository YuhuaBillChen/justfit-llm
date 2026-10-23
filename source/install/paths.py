# Modified for JustFit: product identity and integration.
"""Immutable program files and writable per-user data, for source or release."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
DATA = Path.home() / "Library/Application Support/JustFit" if PACKAGED else ROOT
MODELS = DATA / "models" if PACKAGED else ROOT / "install/models"
RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
BINARY = ROOT / ("engine/justfit-engine" if PACKAGED else "build/justfit-engine")
