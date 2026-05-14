"""
Unified data loaders for WESAD, DEPRESJON, and PSYKOSE.

Each loader exposes:
    .subjects                -> list[str]
    .load(subject_id)        -> dict with keys:
        'subject':   str
        'signals':   dict of np.ndarray (raw signals, varying sample rates)
        'fs':        dict mapping signal name -> sample rate in Hz
        'labels':    dict of per-signal label arrays (or None)
        'meta':      dict with demographics, scores, etc.
    .iter_subjects()         -> generator of the above
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

import config


# ============================================================================
# WESAD
# ============================================================================
class WESADLoader:
    """
    Loads the wrist-only signals from WESAD's per-subject .pkl files.

    Pickle structure (nested dict produced by the WESAD authors):
        data['signal']['wrist']['ACC']  -> (N, 3) float, 32 Hz
        data['signal']['wrist']['BVP']  -> (N, 1) float, 64 Hz
        data['signal']['wrist']['EDA']  -> (N, 1) float,  4 Hz
        data['signal']['wrist']['TEMP'] -> (N, 1) float,  4 Hz
        data['label']                   -> (M,)  int,  700 Hz (chest rate)
        data['subject']                 -> bytes e.g. b'S2'

    The label vector is at 700 Hz. We downsample it via nearest-neighbor
    indexing to match each wrist signal's sample rate.
    """
    def __init__(self, root: Path = config.WESAD_ROOT):
        self.root = Path(root)
        self.subjects = config.WESAD_SUBJECTS

    def _pkl_path(self, sid: str) -> Path:
        return self.root / sid / f"{sid}.pkl"

    def load(self, subject: str) -> dict:
        path = self._pkl_path(subject)
        if not path.exists():
            raise FileNotFoundError(f"WESAD pickle not found: {path}")

        # WESAD pkl was saved under Python 2; latin1 decoding is required.
        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        wrist = raw["signal"]["wrist"]
        labels_700hz = np.asarray(raw["label"]).astype(np.int8)

        # Squeeze single-channel signals from (N, 1) to (N,)
        bvp  = np.asarray(wrist["BVP"]).squeeze().astype(np.float32)
        eda  = np.asarray(wrist["EDA"]).squeeze().astype(np.float32)
        temp = np.asarray(wrist["TEMP"]).squeeze().astype(np.float32)
        acc  = np.asarray(wrist["ACC"]).astype(np.float32) / 64.0  # (N, 3) in g

        signals = {"ACC": acc, "BVP": bvp, "EDA": eda, "TEMP": temp}
        fs = config.WESAD_FS_WRIST

        # Resample the 700 Hz label vector down to each wrist sample rate.
        labels_by_signal = {}
        for name, arr in signals.items():
            n_samples = arr.shape[0]
            idx = np.linspace(0, len(labels_700hz) - 1, n_samples).astype(np.int64)
            labels_by_signal[name] = labels_700hz[idx]

        # Path to questionnaire (we don't parse it here; just store the path)
        quest_path = self.root / subject / f"{subject}_quest.csv"
        meta = {"subject": subject}
        if quest_path.exists():
            meta["quest_csv"] = str(quest_path)

        return {
            "subject": subject,
            "signals": signals,
            "fs": fs,
            "labels_700hz": labels_700hz,
            "labels": labels_by_signal,
            "meta": meta,
        }

    def iter_subjects(self) -> Iterator[dict]:
        for sid in self.subjects:
            yield self.load(sid)


# ============================================================================
# DEPRESJON
# ============================================================================
class DepresjonLoader:
    """
    DEPRESJON per-subject CSV format:
        timestamp,date,activity
        2003-05-07 12:00:00,2003-05-07,143
        ...
    One row per minute. `activity` is an integer activity count.

    scores.csv columns:
        number,days,gender,age,afftype,melanch,inpatient,edu,
        marriage,work,madrs1,madrs2
    where `number` is e.g. 'condition_1' or 'control_1'.
    """
    def __init__(self, root: Path = config.DEPRESJON_ROOT):
        self.root = Path(root)
        self.condition_dir = self.root / "condition"
        self.control_dir   = self.root / "control"
        self.scores_path   = self.root / "scores.csv"

        self.condition_subjects = sorted(
            [p.stem for p in self.condition_dir.glob("condition_*.csv")],
            key=lambda s: int(s.split("_")[1]),
        )
        self.control_subjects = sorted(
            [p.stem for p in self.control_dir.glob("control_*.csv")],
            key=lambda s: int(s.split("_")[1]),
        )
        self.subjects = self.condition_subjects + self.control_subjects

        self._scores = pd.read_csv(self.scores_path) if self.scores_path.exists() else None

    def _path_for(self, subject: str) -> Path:
        if subject.startswith("condition_"):
            return self.condition_dir / f"{subject}.csv"
        if subject.startswith("control_"):
            return self.control_dir / f"{subject}.csv"
        raise ValueError(f"Unknown DEPRESJON subject id: {subject}")

    def load(self, subject: str) -> dict:
        path = self._path_for(subject)
        df = pd.read_csv(path, parse_dates=["timestamp"])

        activity = df["activity"].to_numpy(dtype=np.float32)
        timestamps = df["timestamp"].to_numpy()

        is_patient = subject.startswith("condition_")
        meta = {"subject": subject, "is_patient": is_patient}

        if self._scores is not None:
            row = self._scores[self._scores["number"] == subject]
            if len(row) == 1:
                meta.update(row.iloc[0].to_dict())

        label = 1 if is_patient else 0
        labels = np.full(activity.shape, label, dtype=np.int8)

        return {
            "subject": subject,
            "signals": {"ACTIVITY": activity},
            "timestamps": timestamps,
            "fs": {"ACTIVITY": 1.0 / config.DEPRESJON_SAMPLE_PERIOD_S},  # Hz
            "labels": {"ACTIVITY": labels},
            "subject_label": label,
            "meta": meta,
        }

    def iter_subjects(self) -> Iterator[dict]:
        for sid in self.subjects:
            yield self.load(sid)


# ============================================================================
# PSYKOSE
# ============================================================================
class PsykoseLoader:
    """
    PSYKOSE per-subject CSV uses the same format as DEPRESJON:
        timestamp,date,activity
    """
    def __init__(self, root: Path = config.PSYKOSE_ROOT):
        self.root = Path(root)
        self.patient_dir = self.root / "patient"
        self.control_dir = self.root / "control"
        self.info_path   = self.root / "patients_info.csv"
        self.days_path   = self.root / "days.csv"

        self.patient_subjects = sorted(
            [p.stem for p in self.patient_dir.glob("patient_*.csv")],
            key=lambda s: int(s.split("_")[1]),
        )
        self.control_subjects = sorted(
            [p.stem for p in self.control_dir.glob("control_*.csv")],
            key=lambda s: int(s.split("_")[1]),
        )
        self.subjects = self.patient_subjects + self.control_subjects

        self._info = pd.read_csv(self.info_path) if self.info_path.exists() else None
        self._days = pd.read_csv(self.days_path) if self.days_path.exists() else None

    def _path_for(self, subject: str) -> Path:
        if subject.startswith("patient_"):
            return self.patient_dir / f"{subject}.csv"
        if subject.startswith("control_"):
            return self.control_dir / f"{subject}.csv"
        raise ValueError(f"Unknown PSYKOSE subject id: {subject}")

    def load(self, subject: str) -> dict:
        path = self._path_for(subject)
        df = pd.read_csv(path, parse_dates=["timestamp"])

        activity = df["activity"].to_numpy(dtype=np.float32)
        timestamps = df["timestamp"].to_numpy()

        is_patient = subject.startswith("patient_")
        meta = {"subject": subject, "is_patient": is_patient}

        # patients_info.csv key column varies between releases; try a few.
        if self._info is not None:
            for key_col in ("userid", "id", "number", "user", "subject"):
                if key_col in self._info.columns:
                    row = self._info[self._info[key_col].astype(str) == subject]
                    if len(row) == 1:
                        meta.update(row.iloc[0].to_dict())
                    break

        label = 1 if is_patient else 0
        labels = np.full(activity.shape, label, dtype=np.int8)

        return {
            "subject": subject,
            "signals": {"ACTIVITY": activity},
            "timestamps": timestamps,
            "fs": {"ACTIVITY": 1.0 / config.PSYKOSE_SAMPLE_PERIOD_S},
            "labels": {"ACTIVITY": labels},
            "subject_label": label,
            "meta": meta,
        }

    def iter_subjects(self) -> Iterator[dict]:
        for sid in self.subjects:
            yield self.load(sid)