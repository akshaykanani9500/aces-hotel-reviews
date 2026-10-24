"""Hotel Lab: an ACES-style test bench for how AI agents read hotel-room reviews.

Built on top of ACES (Allouah et al., 2025, arXiv:2508.02630, github.com/mycustomai/ACES).
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
STIMULI_DIR = PACKAGE_DIR / "stimuli"
CONFIG_DIR = PACKAGE_DIR / "config"
RUNS_DIR = REPO_ROOT / "runs"

__version__ = "0.1.0"
