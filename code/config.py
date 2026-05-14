"""
Central configuration for paths and dataset constants.
Everything else imports from here, so paths are never hardcoded elsewhere.
"""
from pathlib import Path

# Project root: W:\Projects\wearable_project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset roots — match the folder layout you have on disk.
WESAD_ROOT     = PROJECT_ROOT / "WESAD"
DEPRESJON_ROOT = PROJECT_ROOT / "depresjon" / "data"
PSYKOSE_ROOT   = PROJECT_ROOT / "psykose"

# Output dirs — created automatically.
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
for d in (FEATURES_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# WESAD constants — from the official readme
# ----------------------------------------------------------------------------
# Note: S1 and S12 are intentionally absent from the release.
WESAD_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                  "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

# Sampling rates of the wrist-worn Empatica E4 in WESAD (Hz).
# These match the target ESP32 hardware sampling rates exactly.
WESAD_FS_WRIST = {
    "ACC":  32,
    "BVP":  64,
    "EDA":   4,
    "TEMP":  4,
}

# Label codes from the WESAD readme.
# Labels 5/6/7 are protocol-internal transitions and are ignored.
WESAD_LABEL_MAP = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore_5",
    6: "ignore_6",
    7: "ignore_7",
}
WESAD_LABELS_OF_INTEREST = {1, 2, 3}  # baseline, stress, amusement

# ----------------------------------------------------------------------------
# DEPRESJON / PSYKOSE constants
# ----------------------------------------------------------------------------
# Both use Actiwatch AW4 wrist actigraph with per-minute activity counts.
DEPRESJON_SAMPLE_PERIOD_S = 60.0   # one sample per minute
PSYKOSE_SAMPLE_PERIOD_S   = 60.0

# ----------------------------------------------------------------------------
# Sanity check at import time
# ----------------------------------------------------------------------------
def _check_paths():
    missing = []
    for name, p in [("WESAD", WESAD_ROOT),
                    ("DEPRESJON", DEPRESJON_ROOT),
                    ("PSYKOSE", PSYKOSE_ROOT)]:
        if not p.exists():
            missing.append(f"  {name}: {p}")
    if missing:
        print("WARNING: missing dataset roots:\n" + "\n".join(missing))

_check_paths()