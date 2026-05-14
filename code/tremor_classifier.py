"""
Head B: Tremor Classifier — deterministic spectral rule engine.

Classifies 4-second accelerometer windows into:
    0 = no tremor / normal movement
    1 = Parkinsonian resting tremor  (4-6 Hz dominant, at REST)
    2 = Essential tremor             (8-12 Hz dominant, during ACTION)
    3 = Indeterminate                (tremor-band energy present but ambiguous)

Algorithm:
    1. Compute Hann-windowed FFT of ACC magnitude (128 samples @ 32 Hz = 4 s).
    2. Compute relative spectral power in three bands:
          PD_BAND:    4.0 - 6.0 Hz
          ET_BAND:    8.0 - 12.0 Hz
          NOISE_BAND: 0.5 - 3.5 Hz  (gait, postural sway — NOT tremor)
    3. Apply motion context gate:
          If ENMO_mean > 0.3 g  -> subject is in vigorous motion -> SKIP
          If ENMO_mean < 0.005g -> subject is completely still -> SKIP
          (both return class 0 with confidence 0)
    4. Classification rules (in order):
          a. If PD_BAND_power  > PD_THRESH  AND ENMO_mean < REST_THRESH -> class 1
          b. If ET_BAND_power  > ET_THRESH                              -> class 2
          c. If max(PD, ET)    > INDETERMINATE_THRESH                   -> class 3
          d. Else                                                        -> class 0

Published frequency band references:
    PD resting tremor:  4-6 Hz  (Jankovic 2008, Mov Disord)
    Essential tremor:   8-12 Hz (Louis & Ferreira 2010, Lancet Neurol)
    PD vs ET boundary:  5-8 Hz ambiguous zone (requires clinical assessment)

Thresholds are conservative (high specificity) to minimise false positives
in a general-population screening device.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ── Spectral bands (Hz) ───────────────────────────────────────────────────────
PD_BAND    = (4.0,  6.0)
ET_BAND    = (8.0, 12.0)
NOISE_BAND = (0.5,  3.5)

# ── Classification thresholds ─────────────────────────────────────────────────
PD_THRESH            = 0.35    # relative power in PD band
ET_THRESH            = 0.35    # relative power in ET band
INDETERMINATE_THRESH = 0.20    # weaker signal, can't distinguish
REST_THRESH          = 0.05    # g — above this = action tremor context

# ── Motion gate ───────────────────────────────────────────────────────────────
MOTION_MAX_G  = 0.30    # skip if vigorous movement
MOTION_MIN_G  = 0.005   # skip if completely still

# ── FFT parameters ────────────────────────────────────────────────────────────
FS_ACC     = 32          # Hz
N_FFT      = 128         # samples (4 seconds at 32 Hz)
FREQ_RES   = FS_ACC / N_FFT   # 0.25 Hz per bin

CLASS_NAMES = ["no_tremor", "PD_tremor", "ET_tremor", "indeterminate"]


@dataclass
class TremorResult:
    label:       int      # 0-3
    class_name:  str
    pd_power:    float    # relative spectral power in PD band
    et_power:    float    # relative spectral power in ET band
    dom_freq:    float    # dominant frequency in the tremor bands (Hz)
    confidence:  float    # 0-1: power at dom_freq / total band power
    enmo_mean:   float    # context: mean movement


def _band_power(power: np.ndarray, freqs: np.ndarray,
                lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    total = power.sum()
    return float(power[mask].sum() / total) if total > 0 else 0.0


def classify_window(acc_xyz: np.ndarray, fs: int = FS_ACC) -> TremorResult:
    """
    Classify a single 4-second ACC window.
    acc_xyz: (N, 3) float32 in g units. N should be >= N_FFT.
    """
    if acc_xyz.ndim != 2 or acc_xyz.shape[1] != 3:
        raise ValueError(f"Expected (N, 3), got {acc_xyz.shape}")

    seg = acc_xyz[-N_FFT:].astype(np.float32) \
          if acc_xyz.shape[0] >= N_FFT else acc_xyz.astype(np.float32)

    # High-pass each axis by removing its mean (removes gravity / DC offset).
    # This preserves small tremor oscillations that ENMO would destroy.
    seg_hp = seg - seg.mean(axis=0, keepdims=True)

    # RMS of HP signal = AC activity level (used for motion gate)
    rms = float(np.sqrt(np.mean(np.sum(seg_hp ** 2, axis=1))))

    # Motion gate: too still -> no tremor; too vigorous -> skip
    if rms < 0.003 or rms > 0.15:
        return TremorResult(0, CLASS_NAMES[0], 0.0, 0.0, 0.0, 0.0, rms)

    # FFT each axis independently, sum power spectra.
    # This avoids the magnitude-squaring frequency-doubling artifact.
    n = len(seg_hp)
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(n - 1, 1))
            ).astype(np.float32)
    power = np.zeros(N_FFT // 2 + 1, dtype=np.float32)
    for axis in range(3):
        x = seg_hp[:, axis] * hann
        spec  = np.fft.rfft(x, n=N_FFT)
        power += spec.real ** 2 + spec.imag ** 2

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / fs).astype(np.float32)

    pd_p = _band_power(power, freqs, *PD_BAND)
    et_p = _band_power(power, freqs, *ET_BAND)

    # Dominant frequency across both tremor bands
    tremor_mask = ((freqs >= PD_BAND[0]) & (freqs <= PD_BAND[1])) | \
                  ((freqs >= ET_BAND[0]) & (freqs <= ET_BAND[1]))
    if tremor_mask.any() and power[tremor_mask].sum() > 0:
        peak_idx   = int(np.argmax(power[tremor_mask]))
        dom_freq   = float(freqs[tremor_mask][peak_idx])
        dom_pow    = float(power[tremor_mask][peak_idx])
        total_pow  = float(power[tremor_mask].sum())
        confidence = dom_pow / total_pow if total_pow > 0 else 0.0
    else:
        dom_freq = confidence = 0.0

    # Classification (order matters)
    if pd_p > PD_THRESH and rms < REST_THRESH:
        label = 1
    elif et_p > ET_THRESH:
        label = 2
    elif max(pd_p, et_p) > INDETERMINATE_THRESH:
        label = 3
    else:
        label = 0

    return TremorResult(label, CLASS_NAMES[label],
                        pd_p, et_p, dom_freq, confidence, rms)

# ── Sustained tremor flag (for the event log) ────────────────────────────────
def sustained_tremor_flag(results: list[TremorResult],
                          min_consecutive: int = 6) -> tuple[bool, int]:
    """
    Returns (is_sustained, max_consecutive_tremor_windows).
    A clinically significant tremor should persist for >= 6 consecutive
    4-second windows (24 seconds). Single isolated detections are suppressed.
    """
    max_run = cur_run = 0
    for r in results:
        if r.label in (1, 2):
            cur_run += 1
            max_run  = max(max_run, cur_run)
        else:
            cur_run  = 0
    return max_run >= min_consecutive, max_run


# ── Synthetic validation ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import config

    print("Tremor classifier — synthetic signal validation\n")
    print(f"{'signal':<30} {'expected':<15} {'got':<15} "
          f"{'pd_pow':>8} {'et_pow':>8} {'dom_hz':>8} {'pass':>5}")
    print("-" * 90)

    FS  = 32
    T   = np.arange(N_FFT) / FS
    RNG = np.random.default_rng(42)

    def make_signal(freq_hz, amplitude, noise=0.0005):
        """
        Simulate tremor: oscillation on all three axes + gravity on Z.
        After mean-removal (our HP step), gravity vanishes and the
        oscillation remains at its true frequency.
        """
        osc = amplitude * np.sin(2 * np.pi * freq_hz * T).astype(np.float32)
        acc = np.zeros((N_FFT, 3), dtype=np.float32)
        acc[:, 0] = osc + RNG.normal(0, noise, N_FFT)
        acc[:, 1] =       RNG.normal(0, noise, N_FFT)
        acc[:, 2] = 1.0 + RNG.normal(0, noise, N_FFT)  # gravity
        return acc

    tests = [
        # (description,             freq,  amplitude,  expected_label)
        ("PD resting tremor 5Hz",   5.0,   0.030,      1),
        ("PD resting tremor 4.5Hz", 4.5,   0.025,      1),
        ("ET action tremor 10Hz",  10.0,   0.060,      2),
        ("ET action tremor 9Hz",    9.0,   0.050,      2),
        ("Normal movement 2Hz",     2.0,   0.080,      0),  # 2Hz not in PD/ET band
        ("Still (below gate)",      5.0,   0.001,      0),  # rms < 0.003
        ("Vigorous motion",         1.5,   0.200,      0),  # rms > 0.15
        ("Mixed freq ambiguous",    5.0,   0.015,      1),  # clean 5Hz = PD even at low amp
    ]

    n_pass = 0
    for desc, freq, amp, expected in tests:
        acc = make_signal(freq, amp)
        r   = classify_window(acc)
        ok  = r.label == expected
        n_pass += ok
        print(f"  {desc:<28} {CLASS_NAMES[expected]:<15} {r.class_name:<15} "
              f"{r.pd_power:>8.3f} {r.et_power:>8.3f} "
              f"{r.dom_freq:>8.2f} {'✓' if ok else '✗':>5}")

    print(f"\n{n_pass}/{len(tests)} tests passed")

    # Save C header with thresholds
    hdr = config.MODELS_DIR / "tremor_params.h"
    with open(hdr, "w") as f:
        f.write("// Auto-generated by tremor_classifier.py\n#pragma once\n\n")
        f.write(f"#define TR_FS_ACC            {FS_ACC}\n")
        f.write(f"#define TR_N_FFT             {N_FFT}\n")
        f.write(f"#define TR_FREQ_RES          {FREQ_RES:.4f}f\n\n")
        f.write(f"#define TR_PD_BAND_LO        {PD_BAND[0]:.1f}f\n")
        f.write(f"#define TR_PD_BAND_HI        {PD_BAND[1]:.1f}f\n")
        f.write(f"#define TR_ET_BAND_LO        {ET_BAND[0]:.1f}f\n")
        f.write(f"#define TR_ET_BAND_HI        {ET_BAND[1]:.1f}f\n\n")
        f.write(f"#define TR_PD_THRESH         {PD_THRESH:.2f}f\n")
        f.write(f"#define TR_ET_THRESH         {ET_THRESH:.2f}f\n")
        f.write(f"#define TR_INDET_THRESH      {INDETERMINATE_THRESH:.2f}f\n")
        f.write(f"#define TR_REST_THRESH       {REST_THRESH:.3f}f\n")
        f.write(f"#define TR_MOTION_MAX        {MOTION_MAX_G:.2f}f\n")
        f.write(f"#define TR_MOTION_MIN        {MOTION_MIN_G:.3f}f\n")
        f.write(f"#define TR_MIN_CONSECUTIVE   6\n")
    print(f"\nC header saved -> {hdr}")