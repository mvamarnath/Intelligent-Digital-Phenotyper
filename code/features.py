"""
features.py - v1

Feature extractor for wrist-worn physiological signals.
Designed to run identically on Windows (NumPy) and ESP32-S3 (C/C++).

CAVEAT - HRV reliability:
    RMSSD and SDNN are computed from wrist BVP. Published analyses of
    WESAD (e.g. Schmidt et al. 2018) show that wrist-PPG-derived HRV does
    not reliably discriminate stress from baseline at this sampling rate
    (64 Hz, ~15.6 ms IBI resolution). We retain these features in the
    output vector with NaN-on-motion gating, but downstream models should
    treat them as low-reliability signals. Primary stress discriminators
    on the wrist are EDA (SCL, SCR_count) and skin temperature.

Feature order (fixed - do not reorder, the autoencoder depends on this):
    0  ENMO_mean        movement intensity (g)
    1  ENMO_std         movement variability (g)
    2  DOM_FREQ         dominant accelerometer frequency, NaN if still (Hz)
    3  DOM_FREQ_POWER   relative power at dominant frequency (0-1)
    4  HR_MEAN          heart rate (BPM)
    5  RMSSD            cardiac variability (ms) [low reliability on wrist]
    6  SDNN             cardiac variability (ms) [low reliability on wrist]
    7  SCL              tonic skin conductance level (uS)
    8  SCR_COUNT        phasic skin conductance responses (count/min)
    9  TEMP_MEAN        skin temperature (degC)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ============================================================================
# Constants
# ============================================================================
WINDOW_SECONDS = 60                  # one feature vector per 60 s
FEATURE_NAMES = (
    "ENMO_mean", "ENMO_std",
    "DOM_FREQ", "DOM_FREQ_POWER",
    "HR_MEAN", "RMSSD", "SDNN",
    "SCL", "SCR_COUNT",
    "TEMP_MEAN",
)
N_FEATURES = len(FEATURE_NAMES)

# FFT window for dominant-frequency estimation:
# 2 seconds at 32 Hz = 64 samples, gives 0.5 Hz bin resolution.
# We use 4 seconds = 128 samples for 0.25 Hz resolution, which matters
# at the 4-8 Hz boundary between Parkinsonian (4-6 Hz) and essential (8-12 Hz)
# tremor. Both window sizes are powers of two for radix-2 FFT.
FFT_WINDOW_S = 4
FFT_WINDOW_HZ_RES = 1.0 / FFT_WINDOW_S   # 0.25 Hz

# BVP/PPG band of interest: 0.5-4 Hz covers 30-240 BPM
BVP_BAND_LOW_HZ  = 0.5
BVP_BAND_HIGH_HZ = 4.0

# EDA tonic/phasic split frequency
EDA_TONIC_CUTOFF_HZ = 0.05


# ============================================================================
# Lightweight IIR filters
# ----------------------------------------------------------------------------
# We use 1st-order single-pole IIR (one-zero, one-pole) for simplicity and
# portability. These are NOT Butterworth; they are direct exponential
# smoothers. The transfer function is straightforward to implement in C:
#     y[n] = alpha * x[n] + (1 - alpha) * y[n-1]      (low-pass)
#     y[n] = x[n] - x[n-1] + (1 - alpha) * y[n-1]     (high-pass, derived)
#
# For bandpass we cascade a high-pass and a low-pass. This is less sharp
# than Butterworth but adequate for our feature extraction and matches
# what the ESP32 will compute.
# ============================================================================
def _alpha_from_cutoff(fc_hz: float, fs_hz: float) -> float:
    """RC-style alpha for a first-order IIR. fc is the -3 dB cutoff."""
    rc = 1.0 / (2.0 * np.pi * fc_hz)
    dt = 1.0 / fs_hz
    return float(dt / (rc + dt))


def lowpass_1pole(x: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """First-order low-pass. Forward direction only (causal, like the MCU)."""
    a = _alpha_from_cutoff(fc_hz, fs_hz)
    y = np.empty_like(x, dtype=np.float32)
    if x.size == 0:
        return y
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def highpass_1pole(x: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """First-order high-pass. y[n] = a*(y[n-1] + x[n] - x[n-1]).

    Note: the alpha here is the *complement* of the low-pass alpha,
    derived from the bilinear transform of the analog RC high-pass.
    """
    rc = 1.0 / (2.0 * np.pi * fc_hz)
    dt = 1.0 / fs_hz
    a = float(rc / (rc + dt))
    y = np.empty_like(x, dtype=np.float32)
    if x.size == 0:
        return y
    y[0] = 0.0
    for i in range(1, x.size):
        y[i] = a * (y[i - 1] + x[i] - x[i - 1])
    return y


def bandpass_1pole(x: np.ndarray, lo_hz: float, hi_hz: float, fs_hz: float) -> np.ndarray:
    """Cascade: high-pass at lo_hz, then low-pass at hi_hz."""
    return lowpass_1pole(highpass_1pole(x, lo_hz, fs_hz), hi_hz, fs_hz)


# ============================================================================
# Accelerometer features
# ============================================================================
def enmo(acc_xyz: np.ndarray) -> np.ndarray:
    """
    Euclidean Norm Minus One. Input shape (N, 3) in g units.
    Returns shape (N,) clipped at 0 (negative values come from sensor noise
    and don't represent real movement).

    ENMO is the standard accelerometer activity metric in the wearables
    literature (van Hees et al.) because it is orientation-invariant and
    rejects the static 1g gravity vector.
    """
    if acc_xyz.ndim != 2 or acc_xyz.shape[1] != 3:
        raise ValueError(f"acc_xyz must be (N, 3), got {acc_xyz.shape}")
    mag = np.sqrt(np.sum(acc_xyz.astype(np.float32) ** 2, axis=1))
    return np.clip(mag - 1.0, 0.0, None).astype(np.float32)


def dominant_frequency(acc_mag: np.ndarray, fs_hz: int) -> tuple[float, float]:
    """
    Dominant frequency of accelerometer magnitude in the last FFT_WINDOW_S
    of data, computed via radix-2 FFT (Hann-windowed).

    Returns (freq_hz, relative_power) where relative_power is the power at
    the dominant bin divided by total power in the 0.5-16 Hz band.

    If the window has insufficient energy (subject is still), returns (0.0, 0.0)
    to indicate "no detectable rhythmic motion."
    """
    n = FFT_WINDOW_S * fs_hz
    if acc_mag.size < n:
        return (float("nan"), float("nan"))

    seg = acc_mag[-n:].astype(np.float32)
    seg = seg - seg.mean()                       # remove DC
    if np.all(seg == 0):
        return (0.0, 0.0)

    # Hann window in float32 (precomputable on the MCU)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1)).astype(np.float32)
    seg = seg * w

    # rfft is fine on the host; on the MCU we'll use ESP-DSP's real radix-2.
    spec = np.fft.rfft(seg)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz).astype(np.float32)

    # Look only in the 0.5 - 16 Hz band (excludes DC drift and >Nyquist noise)
    band = (freqs >= 0.5) & (freqs <= min(16.0, fs_hz / 2.0))
    if not band.any() or power[band].sum() == 0:
        return (0.0, 0.0)

    band_power = power[band]
    band_freqs = freqs[band]
    peak_idx = int(np.argmax(band_power))
    peak_freq = float(band_freqs[peak_idx])
    rel_power = float(band_power[peak_idx] / band_power.sum())
    return (peak_freq, rel_power)


# ============================================================================
# PPG / BVP -> heart rate features
# ============================================================================
def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Causal moving average via cumulative sum. Output same length as x.

    For samples earlier than n, the window expands from the start (no padding,
    no edge artifacts). Matches what a streaming MCU implementation would do
    after its first n samples have arrived.
    """
    if n <= 1 or x.size == 0:
        return x.astype(np.float32, copy=True)
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.size):
        lo = max(0, i - n + 1)
        out[i] = float((csum[i + 1] - csum[lo]) / (i + 1 - lo))
    return out


def detect_bvp_peaks(bvp: np.ndarray, fs_hz: int) -> np.ndarray:
    """
    PPG peak detector based on Elgendi et al. (2013).

    Two-moving-average block detection:
      1. Bandpass 0.5-8 Hz to keep the full pulse waveform.
      2. Clip negatives to zero (we only care about systolic upstrokes).
      3. Square -> emphasizes peaks, suppresses baseline.
      4. Compute short MA over ~111 ms (typical systolic peak width).
      5. Compute long MA over ~667 ms (typical beat-to-beat duration).
      6. A "block of interest" is a contiguous run where short_MA > long_MA.
      7. Reject blocks shorter than the expected peak width.
      8. Within each surviving block, the peak is argmax of the filtered signal.
      9. Enforce a 300 ms refractory between accepted peaks.

    Returns peak sample indices (int64).

    Reference:
      Elgendi M, et al. "Systolic Peak Detection in Acceleration
      Photoplethysmograms Measured from Emergency Responders in Tropical
      Conditions." PLOS ONE 8(10): e76585, 2013.
    """
    if bvp.size < fs_hz * 3:
        return np.array([], dtype=np.int64)

    # 1. Bandpass 0.5-8 Hz (wider than before to keep waveform shape).
    filt = bandpass_1pole(bvp.astype(np.float32),
                          BVP_BAND_LOW_HZ, 8.0, fs_hz)

    # 2 + 3. Clip to positive, then square.
    clipped = np.clip(filt, 0.0, None)
    sq = clipped * clipped

    # 4 + 5. Two moving averages.
    w_peak = max(1, int(round(0.111 * fs_hz)))   # ~111 ms
    w_beat = max(1, int(round(0.667 * fs_hz)))   # ~667 ms
    ma_peak = _moving_average(sq, w_peak)
    ma_beat = _moving_average(sq, w_beat)

    # Threshold rises with overall signal strength (offset prevents firing
    # on near-zero baseline; alpha=0 is the classic Elgendi setting).
    threshold = ma_beat  # + 0  (alpha=0)

    # 6. Blocks of interest = ma_peak > threshold.
    above = ma_peak > threshold

    # 7-8. Walk through contiguous "above" blocks.
    peaks: list[int] = []
    refractory = int(0.30 * fs_hz)
    last_accepted = -refractory * 2
    min_block_len = w_peak  # block must be at least one peak-width long

    n = above.size
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        # Find end of this block
        j = i
        while j < n and above[j]:
            j += 1
        block_len = j - i
        if block_len >= min_block_len:
            # Argmax of the FILTERED signal within the block
            local = filt[i:j]
            peak_idx = i + int(np.argmax(local))
            if peak_idx - last_accepted >= refractory:
                peaks.append(peak_idx)
                last_accepted = peak_idx
        i = j

    return np.asarray(peaks, dtype=np.int64)


def hr_features(bvp: np.ndarray, fs_hz: int,
                acc_mag: np.ndarray | None = None,
                acc_fs_hz: int | None = None) -> tuple[float, float, float]:
    """
    Returns (mean_HR_bpm, RMSSD_ms, SDNN_ms) from BVP.

    Signal-quality gate (NEW):
        If a same-window accelerometer magnitude is provided AND its
        standard deviation exceeds 0.05 g, the window is considered
        motion-corrupted and all three values are returned as NaN.
        0.05 g is the threshold used in van Hees et al. for distinguishing
        sustained motion from postural sway.

    IBI filtering:
        Stage 1: drop IBIs outside [300, 2000] ms.
        Stage 2: drop IBIs that deviate by >15% from a 5-beat local median
                 (Berntson-style outlier filter).

    Returns NaN if fewer than 10 clean IBIs survive filtering.
    """
    # Motion gate
    if acc_mag is not None and acc_mag.size > 0:
        motion_std = float(acc_mag.std(ddof=0))
        if motion_std > 0.05:
            return (float("nan"), float("nan"), float("nan"))

    peaks = detect_bvp_peaks(bvp, fs_hz)
    if peaks.size < 10:
        return (float("nan"), float("nan"), float("nan"))

    ibi_ms = np.diff(peaks).astype(np.float32) * (1000.0 / fs_hz)

    # Stage 1: physiological plausibility
    plausible = (ibi_ms >= 300.0) & (ibi_ms <= 2000.0)
    ibi_ms = ibi_ms[plausible]
    if ibi_ms.size < 10:
        return (float("nan"), float("nan"), float("nan"))

    # Stage 2: local-median outlier filter, tightened to 15%
    n = ibi_ms.size
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - 2)
        hi = min(n, i + 3)
        local_median = np.median(ibi_ms[lo:hi])
        if local_median <= 0:
            keep[i] = False
            continue
        rel_dev = abs(ibi_ms[i] - local_median) / local_median
        if rel_dev > 0.15:
            keep[i] = False
    ibi_clean = ibi_ms[keep]

    if ibi_clean.size < 5:
        return (float("nan"), float("nan"), float("nan"))

    mean_hr = 60000.0 / float(ibi_clean.mean())

    # RMSSD over genuinely consecutive surviving IBIs
    surviving_indices = np.where(keep)[0]
    consecutive_pairs = np.diff(surviving_indices) == 1
    if not consecutive_pairs.any():
        rmssd = float("nan")
    else:
        diffs = np.diff(ibi_ms[keep])
        diffs = diffs[consecutive_pairs]
        rmssd = float(np.sqrt(np.mean(diffs ** 2)))

    sdnn = float(ibi_clean.std(ddof=0))
    return (mean_hr, rmssd, sdnn)

# ============================================================================
# EDA features
# ============================================================================
def eda_features(eda: np.ndarray, fs_hz: int) -> tuple[float, int]:
    """
    Returns (SCL_uS, SCR_count) from electrodermal activity.

    SCL  = tonic skin conductance level (slow component), uS.
    SCR_count = number of phasic skin conductance responses in this window.

    SCR detection is a derivative-zero-crossing scheme:
      1. Low-pass below 0.05 Hz -> tonic (slow drift)
      2. Subtract tonic from raw -> phasic component
      3. Compute first derivative of phasic
      4. Count positive-going zero crossings with amplitude > 0.05 uS
         (Society for Psychophysiological Research convention)
    """
    if eda.size < fs_hz * 2:
        return (float("nan"), 0)

    x = eda.astype(np.float32)
    tonic = lowpass_1pole(x, EDA_TONIC_CUTOFF_HZ, fs_hz)
    phasic = x - tonic

    scl = float(tonic.mean())

    # Derivative
    dphasic = np.diff(phasic, prepend=phasic[0])
    # Count rising zero-crossings of dphasic where peak amplitude exceeds 0.05 uS
    n_scr = 0
    in_peak = False
    peak_amp = 0.0
    for i in range(1, dphasic.size):
        if dphasic[i - 1] > 0 and dphasic[i] <= 0:
            # local max of phasic at sample i
            if not in_peak:
                peak_amp = float(phasic[i])
                in_peak = True
            else:
                peak_amp = max(peak_amp, float(phasic[i]))
        elif dphasic[i - 1] < 0 and dphasic[i] >= 0 and in_peak:
            # end of a phasic excursion; commit if amplitude exceeds threshold
            if peak_amp > 0.05:
                n_scr += 1
            in_peak = False
            peak_amp = 0.0

    return (scl, n_scr)


# ============================================================================
# Temperature
# ============================================================================
def temp_features(temp: np.ndarray) -> float:
    """Mean skin temperature in degC over the window."""
    if temp.size == 0:
        return float("nan")
    return float(temp.mean())


# ============================================================================
# Top-level: extract one feature vector from a single window
# ============================================================================
@dataclass
class WindowInputs:
    """Per-window slices from each sensor."""
    acc:  np.ndarray   # (N_acc, 3)  @ fs_acc Hz
    bvp:  np.ndarray   # (N_bvp,)    @ fs_bvp Hz
    eda:  np.ndarray   # (N_eda,)    @ fs_eda Hz
    temp: np.ndarray   # (N_temp,)   @ fs_eda Hz (typically same as EDA)
    fs_acc:  int
    fs_bvp:  int
    fs_eda:  int


def extract_one_window(w: WindowInputs) -> np.ndarray:
    """Compute the 10-element feature vector for a single window."""
    out = np.full(N_FEATURES, np.nan, dtype=np.float32)

    # Accelerometer features
    if w.acc.size > 0:
        mag = enmo(w.acc)
        out[0] = float(mag.mean())
        out[1] = float(mag.std(ddof=0))
        # Only report dominant frequency when there is meaningful movement.
        # Below 0.02 g mean ENMO the subject is essentially still and the
        # "dominant frequency" is just sensor noise. Gate to NaN in that case.
        if out[0] >= 0.02:
            f, p = dominant_frequency(mag, w.fs_acc)
            out[2] = f
            out[3] = p
        # else: leave out[2], out[3] as NaN

    # Cardiac features (with motion-artifact gate from accelerometer)
    if w.bvp.size > 0:
        acc_mag = enmo(w.acc) if w.acc.size > 0 else None
        hr, rmssd, sdnn = hr_features(w.bvp, w.fs_bvp,
                                      acc_mag=acc_mag,
                                      acc_fs_hz=w.fs_acc)
        out[4] = hr
        out[5] = rmssd
        out[6] = sdnn

    # EDA features
    if w.eda.size > 0:
        scl, scr = eda_features(w.eda, w.fs_eda)
        out[7] = scl
        out[8] = float(scr)

    # Temperature
    if w.temp.size > 0:
        out[9] = temp_features(w.temp)

    return out


# ============================================================================
# Windowed extraction from a full WESAD recording
# ============================================================================
def extract_wesad_features(signals: dict, fs: dict, labels: dict,
                           window_s: int = WINDOW_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a non-overlapping window over a full WESAD subject recording
    and emit one feature vector per window.

    Window-level label = majority label of the BVP-aligned labels in that
    window. (Could equally pick any aligned signal; BVP is highest-rate
    among single-channel sensors.)

    Returns:
        X: (n_windows, N_FEATURES) float32
        y: (n_windows,) int8
    """
    fs_acc = int(fs["ACC"])
    fs_bvp = int(fs["BVP"])
    fs_eda = int(fs["EDA"])

    # Number of windows: limited by the shortest signal
    n_min_seconds = min(
        signals["ACC"].shape[0]  // fs_acc,
        signals["BVP"].shape[0]  // fs_bvp,
        signals["EDA"].shape[0]  // fs_eda,
        signals["TEMP"].shape[0] // fs_eda,
    )
    n_windows = n_min_seconds // window_s

    X = np.empty((n_windows, N_FEATURES), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.int8)

    bvp_labels = labels["BVP"]

    for i in range(n_windows):
        a0, a1 = i * window_s * fs_acc, (i + 1) * window_s * fs_acc
        b0, b1 = i * window_s * fs_bvp, (i + 1) * window_s * fs_bvp
        e0, e1 = i * window_s * fs_eda, (i + 1) * window_s * fs_eda

        w = WindowInputs(
            acc=signals["ACC"][a0:a1],
            bvp=signals["BVP"][b0:b1],
            eda=signals["EDA"][e0:e1],
            temp=signals["TEMP"][e0:e1],
            fs_acc=fs_acc, fs_bvp=fs_bvp, fs_eda=fs_eda,
        )
        X[i] = extract_one_window(w)

        # Majority label of this window in BVP-aligned label space
        seg = bvp_labels[b0:b1]
        if seg.size:
            vals, counts = np.unique(seg, return_counts=True)
            y[i] = int(vals[np.argmax(counts)])
        else:
            y[i] = 0

    return X, y