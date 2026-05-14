"""
Head E: Sleep/Wake Classifier — van Hees HDCZA algorithm.

No training required. Uses published thresholds from:
    van Hees VT et al. "Estimating sleep parameters using an accelerometer
    without sleep diary." Scientific Reports 2018;8:12975.

Algorithm (per 30-second epoch):
    1. Compute ENMO for each second: mean(||acc|| - 1g, 0) at 32 Hz
       -> 30 values per epoch (one per second, averaged over 32 samples each)
    2. Compute the absolute difference between consecutive 5-second means
       -> ENMO_diff: measure of minute-to-minute movement change
    3. A 30-second epoch is classified SLEEP if:
           ENMO_mean < THRESHOLD_ENMO   (little movement)
        AND ENMO_diff < THRESHOLD_DIFF  (little change in movement)
    4. Apply a 5-epoch median filter to smooth isolated misclassifications.

Published thresholds (wrist, dominant hand):
    THRESHOLD_ENMO = 0.013 g
    THRESHOLD_DIFF = 0.013 g

Outputs per night:
    sleep_onset     : first epoch classified as sleep after 21:00
    wake_time       : last epoch classified as sleep before 12:00 next day
    sleep_duration  : wake_time - sleep_onset (hours)
    sleep_efficiency: fraction of time-in-bed classified as sleep
    midsleep        : midpoint of sleep period (used by Head F)
    n_awakenings    : number of sleep->wake transitions during the night
    fragmentation   : n_awakenings / sleep_duration_hours

Can be used in two modes:
    A. On raw ACC arrays (WESAD / ESP32 deployment)
    B. On per-minute activity counts (DEPRESJON / PSYKOSE)
       — falls back to a simpler threshold on activity count directly.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


# Published van Hees thresholds
THRESHOLD_ENMO = 0.013   # g
THRESHOLD_DIFF = 0.013   # g
EPOCH_S        = 30      # seconds per epoch
SMOOTH_EPOCHS  = 5       # median filter width


@dataclass
class NightSummary:
    date:               str
    sleep_onset_h:      float          # hours from midnight
    wake_time_h:        float
    sleep_duration_h:   float
    sleep_efficiency:   float          # 0-1
    midsleep_h:         float          # hours from midnight
    n_awakenings:       int
    fragmentation:      float          # awakenings / hour
    tib_h:              float          # time in bed (hours)


def classify_epochs_from_acc(acc: np.ndarray, fs: int = 32) -> np.ndarray:
    """
    Input:  acc (N, 3) float32, units = g, at fs Hz
    Output: sleep_epochs (M,) bool, one value per EPOCH_S seconds
            True = sleep, False = wake
    """
    samples_per_epoch = EPOCH_S * fs
    n_epochs = acc.shape[0] // samples_per_epoch

    enmo_per_epoch = np.empty(n_epochs, dtype=np.float32)
    for i in range(n_epochs):
        seg = acc[i * samples_per_epoch:(i + 1) * samples_per_epoch]
        mag = np.sqrt(np.sum(seg ** 2, axis=1))
        enmo_per_epoch[i] = float(np.clip(mag - 1.0, 0, None).mean())

    # 5-second means: fs*5 samples per block
    samples_per_5s = fs * 5
    n_5s = acc.shape[0] // samples_per_5s
    enmo_5s = np.empty(n_5s, dtype=np.float32)
    for i in range(n_5s):
        seg = acc[i * samples_per_5s:(i + 1) * samples_per_5s]
        mag = np.sqrt(np.sum(seg ** 2, axis=1))
        enmo_5s[i] = float(np.clip(mag - 1.0, 0, None).mean())

    # Absolute diff between consecutive 5s means, resampled to epoch rate
    enmo_diff_5s = np.abs(np.diff(enmo_5s, prepend=enmo_5s[0]))
    # Each epoch covers EPOCH_S/5 = 6 five-second blocks
    blocks_per_epoch = EPOCH_S // 5
    n_diff_epochs = n_5s // blocks_per_epoch
    enmo_diff_epoch = np.empty(n_diff_epochs, dtype=np.float32)
    for i in range(n_diff_epochs):
        enmo_diff_epoch[i] = float(
            enmo_diff_5s[i * blocks_per_epoch:(i + 1) * blocks_per_epoch].mean()
        )

    # Align lengths
    n = min(n_epochs, len(enmo_diff_epoch))
    enmo_per_epoch  = enmo_per_epoch[:n]
    enmo_diff_epoch = enmo_diff_epoch[:n]

    raw_sleep = (enmo_per_epoch < THRESHOLD_ENMO) & \
                (enmo_diff_epoch < THRESHOLD_DIFF)

    # 5-epoch median filter
    sleep = _median_filter(raw_sleep, SMOOTH_EPOCHS)
    return sleep


def classify_epochs_from_activity(activity_counts: np.ndarray,
                                  threshold: float = 40.0) -> np.ndarray:
    """
    Fallback for per-minute activity count data (DEPRESJON / PSYKOSE).
    Each minute = one epoch. Sleep = activity < threshold.
    Threshold 40 counts/min is the standard Sadeh/Cole-Kripke value.
    """
    raw_sleep = activity_counts < threshold
    return _median_filter(raw_sleep.astype(bool), 5)


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Causal k-point median filter (no lookahead — matches streaming MCU)."""
    out = np.empty_like(x, dtype=bool)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = bool(np.median(x[lo:i + 1]))
    return out


def summarise_night(sleep_epochs: np.ndarray,
                    timestamps: np.ndarray,
                    epoch_s: float = EPOCH_S,
                    date_str: str = "unknown") -> Optional[NightSummary]:
    """
    Given a boolean sleep array and corresponding timestamps (numpy datetime64),
    extract one NightSummary.

    Looks for the main sleep period between 20:00 and 12:00 next day.
    Returns None if fewer than 60 minutes of sleep found.
    """
    import pandas as pd
    ts = pd.DatetimeIndex(timestamps)
    hours = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

    # Evening window: 20:00 – 12:00 next day (hours > 20 OR hours < 12)
    night_mask = (hours >= 20.0) | (hours < 12.0)
    night_idx  = np.where(night_mask)[0]
    if night_idx.size < 2:
        return None

    night_sleep = sleep_epochs[night_idx]
    if night_sleep.sum() < int(60 * 60 / epoch_s):   # < 60 minutes of sleep
        return None

    # Time in bed: first to last sleep epoch in the night window
    sleep_positions = np.where(night_sleep)[0]
    tib_start = int(sleep_positions[0])
    tib_end   = int(sleep_positions[-1]) + 1

    tib_h      = (tib_end - tib_start) * epoch_s / 3600.0
    onset_h    = float(hours[night_idx[tib_start]])
    wake_h     = float(hours[night_idx[tib_end - 1]]) + epoch_s / 3600.0
    midsleep_h = (onset_h + wake_h) / 2.0

    tib_sleep   = night_sleep[tib_start:tib_end]
    efficiency  = float(tib_sleep.mean())
    sleep_dur_h = efficiency * tib_h

    # Count awakenings (sleep -> wake transitions during TIB)
    transitions  = np.diff(tib_sleep.astype(int))
    n_awakenings = int((transitions == -1).sum())
    fragmentation = n_awakenings / sleep_dur_h if sleep_dur_h > 0 else 0.0

    return NightSummary(
        date=date_str,
        sleep_onset_h=onset_h,
        wake_time_h=wake_h,
        sleep_duration_h=sleep_dur_h,
        sleep_efficiency=efficiency,
        midsleep_h=midsleep_h,
        n_awakenings=n_awakenings,
        fragmentation=fragmentation,
        tib_h=tib_h,
    )


# ── Validation on DEPRESJON (activity counts, all subjects) ──────────────────
if __name__ == "__main__":
    import config
    from data_loaders import DepresjonLoader, PsykoseLoader

    print("Validating sleep classifier on DEPRESJON + PSYKOSE activity counts\n")
    print(f"{'dataset':<12} {'subject':<20} {'label':>5} "
          f"{'nights':>7} {'mean_dur_h':>11} {'mean_eff':>9} "
          f"{'mean_frag':>10}")
    print("-" * 78)

    for LoaderClass, dset_name, label_map in [
        (DepresjonLoader, "depresjon", {"condition": "depression", "control": "control"}),
        (PsykoseLoader,   "psykose",   {"patient": "schizophrenia", "control": "control"}),
    ]:
        loader = LoaderClass()
        # Summarise first 3 subjects per class to keep output readable
        shown = {"condition": 0, "control": 0, "patient": 0}
        limit = 2

        for sid in loader.subjects:
            prefix = sid.split("_")[0]
            if shown.get(prefix, 0) >= limit:
                continue
            shown[prefix] = shown.get(prefix, 0) + 1

            d = loader.load(sid)
            activity   = d["signals"]["ACTIVITY"]
            timestamps = d["timestamps"]
            label_str  = label_map.get(prefix, prefix)

            sleep_epochs = classify_epochs_from_activity(activity)

            # Summarise per night
            import pandas as pd
            ts  = pd.DatetimeIndex(timestamps)
            days = np.unique(ts.date)
            nights = []
            for day in days:
                day_mask = ts.date == day
                s = summarise_night(
                    sleep_epochs[day_mask],
                    timestamps[day_mask],
                    epoch_s=60.0,
                    date_str=str(day),
                )
                if s is not None:
                    nights.append(s)

            if not nights:
                continue

            mean_dur  = np.mean([n.sleep_duration_h  for n in nights])
            mean_eff  = np.mean([n.sleep_efficiency   for n in nights])
            mean_frag = np.mean([n.fragmentation      for n in nights])

            print(f"  {dset_name:<10} {sid:<20} {label_str:>12} "
                  f"{len(nights):>7d} {mean_dur:>11.2f} "
                  f"{mean_eff:>9.3f} {mean_frag:>10.3f}")

    print("\nSleep classifier ready. No training required.")
    print("Parameters: THRESHOLD_ENMO=0.013g, THRESHOLD_DIFF=0.013g, "
          "SMOOTH=5 epochs")