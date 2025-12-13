import os
from pathlib import Path

# Dynamic Paths
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Project Constants
RANDOM_STATE = 42
TARGET_COL = "label" 