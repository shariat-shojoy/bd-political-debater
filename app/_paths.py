"""
Cross-platform path resolution for the BD Political Debater.

All modules should import BASE_DIR + CONFIG_PATH from here instead of
hardcoding paths. This makes the project portable across Linux/Windows/macOS
and any clone location.
"""
from __future__ import annotations

from pathlib import Path

# This file lives at: <project_root>/app/_paths.py
# So the project root is two levels up.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Subdirectories — all relative to the project root.
CONFIG_DIR: Path = BASE_DIR / "config"
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"
AUDIO_DIR: Path = DATA_DIR / "audio"
ANIMATIONS_DIR: Path = DATA_DIR / "animations"
LOGS_DIR: Path = BASE_DIR / "logs"
DOWNLOAD_DIR: Path = BASE_DIR / "download"
ASSETS_DIR: Path = BASE_DIR / "assets"

# The main config file.
CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"

# Ensure critical dirs exist (idempotent — safe to call on import).
for _d in (CONFIG_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR,
           AUDIO_DIR, ANIMATIONS_DIR, LOGS_DIR, DOWNLOAD_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
