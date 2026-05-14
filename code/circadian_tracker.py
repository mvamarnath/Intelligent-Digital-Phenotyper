"""
Head F: Circadian Phase Tracker — Forger (1999) two-process ODE model.

Inputs:  per-night sleep/wake sequence from Head E
Outputs: estimated DLMO (Dim Light Melatonin Onset) in hours from midnight,
         circadian phase Z-score vs personal 30-day baseline

The Forger model simulates the circadian pacemaker using two coupled ODEs
driven by a light input proxy derived from the sleep/wake pattern.
Light proxy: 250 lux during wake, 0 lux during sleep.
(Validated in Lim et al. npj Digital Medicine 2024 and prior work.)

Reference:
    Forger DB, Jewett ME, Kronauer RE. "A simpler model of the human
    circadian pacemaker." J Biol Rhythms 1999;14(6):532-537.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ── Forger model constants (published values) ────────────────────────────────
MU      = 0.13      # amplitude recovery rate
Q       = 1.0 / 3.0
K       = 0.55      # light sensitivity
TAU_C   = 24.2      # intrinsic period (hours)
ALPHA_0 = 0.05      # max light drive
P       = 0.50      # light sensitivity exponent
I_0     = 9500.0    # half-saturation constant (lux)
DT_H    = 1.0 / 60  # integration step = 1 minute (hours)

# CBTmin occurs ~7 h after DLMO (Forger 1999)
DLMO_OFFSET_H = 7.0

# Circadian baseline window
BASELINE_DAYS = 30


@dataclass
class CircadianState:
    """Mutable state of the Forger ODE integrator."""
    x:  float = 0.0     # circadian phase variable
    xc: float = 0.1     # amplitude variable


@dataclass
class NightCircadian:
    date:        str
    dlmo_h:      float      # estimated DLMO hours from midnight
    midsleep_h:  float      # from Head E
    phase_advance: float    # positive = advanced vs personal mean
    phase_z:     float      # Z-score vs personal baseline


def _light_drive(lux: float) -> float:
    """Alpha(I): photic drive as a function of light intensity."""
    return ALPHA_0 * (lux ** P) / ((lux ** P) + I_0 ** P)


def _forger_step(state: CircadianState, lux: float, dt: float) -> None:
    """
    Euler integration of the Forger two-process model for one time step dt (hours).

    dx/dt  = pi/12 * (xc - MU*x*(x^2 + xc^2) + B(x,lux))
    dxc/dt = pi/12 * (-x  - MU*xc*(x^2 + xc^2))

    where B(x, lux) = alpha(lux) * (-Q*xc + alpha(lux)*(1 - P*x/K))
    """
    a  = _light_drive(lux)
    B  = a * (-Q * state.xc + a * (1.0 - (P / K) * state.x))

    r2 = state.x ** 2 + state.xc ** 2
    dx  = (np.pi / 12.0) * (state.xc - MU * state.x * r2 + B)
    dxc = (np.pi / 12.0) * (-state.x - MU * state.xc * r2)

    state.x  += dx  * dt
    state.xc += dxc * dt


def estimate_dlmo(sleep_wake_minutes: np.ndarray,
                  state: CircadianState,
                  lux_wake: float = 250.0,
                  lux_sleep: float = 0.0) -> tuple[float, CircadianState]:
    """
    Drive the Forger model forward through one day (1440 minutes).

    sleep_wake_minutes: bool array length 1440 (True=sleep, False=wake)
    Returns (CBTmin_hour_of_day, updated_state).
    CBTmin is the core body temperature minimum; DLMO = CBTmin - 7 h.
    """
    if len(sleep_wake_minutes) < 1440:
        # Pad with wake if short
        pad = np.zeros(1440, dtype=bool)
        pad[:len(sleep_wake_minutes)] = sleep_wake_minutes
        sleep_wake_minutes = pad

    xc_series = np.empty(1440, dtype=np.float32)
    for minute in range(1440):
        lux = lux_sleep if sleep_wake_minutes[minute] else lux_wake
        _forger_step(state, lux, DT_H)
        xc_series[minute] = state.xc

    # CBTmin = trough of xc
    cbtmin_minute = int(np.argmin(xc_series))
    cbtmin_h = cbtmin_minute / 60.0

    dlmo_h = (cbtmin_h - DLMO_OFFSET_H) % 24.0
    return dlmo_h, state


class CircadianTracker:
    """
    Stateful tracker that maintains:
      - The Forger ODE state (updated nightly)
      - A rolling 30-day DLMO history for personal baseline Z-scoring
    """
    def __init__(self, baseline_days: int = BASELINE_DAYS):
        self.state    = CircadianState()
        self.history: deque[float] = deque(maxlen=baseline_days)
        self.results: list[NightCircadian] = []

    def update(self, sleep_wake_minutes: np.ndarray,
               midsleep_h: float,
               date_str: str = "") -> Optional[NightCircadian]:
        """
        Process one night. Returns NightCircadian if baseline is established
        (>= 7 days of history), else None.
        """
        dlmo_h, self.state = estimate_dlmo(sleep_wake_minutes, self.state)
        self.history.append(dlmo_h)

        if len(self.history) < 7:
            return None   # insufficient baseline

        baseline = np.array(self.history)
        mean_dlmo = float(baseline.mean())
        std_dlmo  = float(baseline.std(ddof=0))

        advance = dlmo_h - mean_dlmo
        # Wrap to [-12, 12] to handle midnight crossing
        advance = (advance + 12) % 24 - 12

        phase_z = advance / std_dlmo if std_dlmo > 0.01 else 0.0

        result = NightCircadian(
            date=date_str,
            dlmo_h=dlmo_h,
            midsleep_h=midsleep_h,
            phase_advance=advance,
            phase_z=phase_z,
        )
        self.results.append(result)
        return result


# ── Validation on DEPRESJON ───────────────────────────────────────────────────
if __name__ == "__main__":
    import config
    from data_loaders import DepresjonLoader
    from sleep_classifier import classify_epochs_from_activity, summarise_night
    import pandas as pd

    print("Circadian tracker validation on DEPRESJON\n")
    print(f"{'subject':<20} {'label':<14} {'nights':>6} "
          f"{'mean_dlmo_h':>12} {'mean_phase_z':>13} "
          f"{'n_delayed(z<-1)':>16} {'n_advanced(z>1)':>16}")
    print("-" * 100)

    loader = DepresjonLoader()
    shown  = {"condition": 0, "control": 0}
    limit  = 3

    for sid in loader.subjects:
        prefix = sid.split("_")[0]
        if shown.get(prefix, 0) >= limit:
            continue
        shown[prefix] += 1

        d          = loader.load(sid)
        activity   = d["signals"]["ACTIVITY"]
        timestamps = d["timestamps"]
        label_str  = "depression" if prefix == "condition" else "control"

        sleep_epochs = classify_epochs_from_activity(activity)
        ts  = pd.DatetimeIndex(timestamps)
        days = np.unique(ts.date)

        tracker = CircadianTracker()
        n_results = 0
        phase_zs  = []

        for day in days:
            day_mask = ts.date == day
            night    = summarise_night(
                sleep_epochs[day_mask], timestamps[day_mask],
                epoch_s=60.0, date_str=str(day)
            )
            if night is None:
                continue

            # Build per-minute sleep/wake for this day (1440 slots)
            sw_day = np.zeros(1440, dtype=bool)
            day_indices = np.where(day_mask)[0]
            for idx in day_indices:
                minute = ts[idx].hour * 60 + ts[idx].minute
                if minute < 1440:
                    sw_day[minute] = sleep_epochs[idx]

            result = tracker.update(sw_day, night.midsleep_h, date_str=str(day))
            if result is not None:
                n_results += 1
                phase_zs.append(result.phase_z)

        if not phase_zs:
            continue

        pz = np.array(phase_zs)
        print(f"  {sid:<20} {label_str:<14} {n_results:>6} "
              f"{np.mean([r.dlmo_h for r in tracker.results]):>12.2f} "
              f"{pz.mean():>13.3f} "
              f"{(pz < -1).sum():>16d} "
              f"{(pz > 1).sum():>16d}")

    print("\nCircadian tracker ready. No training required.")
    print("Model: Forger 1999 ODE, light proxy 250/0 lux, integration step 1 min")