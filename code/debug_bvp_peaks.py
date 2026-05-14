"""
Visualize the BVP peak detector on a clean baseline window from WESAD S2.

Run:    python debug_bvp_peaks.py
Output: reports/debug_bvp_peaks.png

The plot shows three panels:
    1. Raw BVP signal
    2. Filtered BVP + detected peaks (red dots) + expected peak count band
    3. Squared signal + adaptive threshold

We deliberately pick a 30-second window labeled as 'baseline' (label==1)
because we know the subject is calm and the true HR should be ~60-90 BPM,
giving us a clean ground truth to evaluate the detector against.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data_loaders import WESADLoader
from features import (
    bandpass_1pole, _moving_average,
    BVP_BAND_LOW_HZ, BVP_BAND_HIGH_HZ,
    detect_bvp_peaks,
)


def find_clean_baseline_window(bvp, bvp_labels, fs_bvp, duration_s=30):
    """Return (start_idx, end_idx) for a 30s slice with label==1 (baseline)."""
    n = duration_s * fs_bvp
    # Find a stretch where all samples have label 1
    is_base = (bvp_labels == 1).astype(np.int8)
    # Cumulative sum trick: stretch of length n with all 1s sums to n
    csum = np.concatenate(([0], np.cumsum(is_base)))
    for start in range(0, len(is_base) - n, fs_bvp):
        if csum[start + n] - csum[start] == n:
            return start, start + n
    raise RuntimeError("No 30 s clean baseline window found.")


def main():
    loader = WESADLoader()
    d = loader.load("S2")
    bvp = d["signals"]["BVP"]
    fs = d["fs"]["BVP"]
    bvp_labels = d["labels"]["BVP"]

    start, end = find_clean_baseline_window(bvp, bvp_labels, fs, duration_s=30)
    print(f"Selected baseline window: samples {start}-{end} "
          f"({(end - start) / fs:.1f} s)")

    seg = bvp[start:end].astype(np.float32)

    # Reproduce the detector's intermediate signals
    filt = bandpass_1pole(seg, BVP_BAND_LOW_HZ, BVP_BAND_HIGH_HZ, fs)
    sq = filt * filt
    win = int(4 * fs)
    local_mean = _moving_average(sq, win)
    floor = float(np.percentile(sq, 50)) * 0.05 + 1e-9
    threshold = np.maximum(local_mean * 1.5, floor)

    # Run the actual detector on the same segment so peak indices are local
    peaks = detect_bvp_peaks(seg, fs)

    duration_s = (end - start) / fs
    n_peaks = peaks.size
    hr_bpm = 60.0 * n_peaks / duration_s if duration_s > 0 else 0.0
    print(f"Detected {n_peaks} peaks in {duration_s:.1f} s -> HR ~ {hr_bpm:.1f} BPM")
    print(f"  (expected: 18-45 peaks for HR 36-90 BPM)")

    t = np.arange(seg.size) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, seg, lw=0.7, color="black")
    axes[0].set_ylabel("Raw BVP")
    axes[0].set_title(
        f"WESAD S2 baseline 30 s window  -  detector found {n_peaks} peaks "
        f"-> HR ~ {hr_bpm:.1f} BPM"
    )

    axes[1].plot(t, filt, lw=0.7, color="navy", label="bandpass 0.5-4 Hz")
    if peaks.size > 0:
        axes[1].plot(peaks / fs, filt[peaks], "ro", ms=6,
                     label=f"detected peaks (n={peaks.size})")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_ylabel("Filtered BVP")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, sq, lw=0.5, color="purple", label="squared filtered BVP")
    axes[2].plot(t, threshold, lw=1.0, color="orange", label="adaptive threshold")
    axes[2].set_ylabel("Squared signal")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out = config.REPORTS_DIR / "debug_bvp_peaks.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Plot saved -> {out}")

    # Also dump some quick stats so we can reason from numbers
    print("\nFiltered BVP stats:")
    print(f"  min={filt.min():.3f}  max={filt.max():.3f}  "
          f"std={filt.std():.3f}  rms={np.sqrt((filt**2).mean()):.3f}")
    print("\nSquared signal stats:")
    print(f"  median={np.median(sq):.3f}  "
          f"p95={np.percentile(sq, 95):.3f}  "
          f"p99={np.percentile(sq, 99):.3f}")
    print("\nThreshold stats:")
    print(f"  min={threshold.min():.3f}  "
          f"median={np.median(threshold):.3f}  "
          f"max={threshold.max():.3f}")


if __name__ == "__main__":
    main()