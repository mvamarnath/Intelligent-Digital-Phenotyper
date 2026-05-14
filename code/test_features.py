"""
Smoke-test the feature extractor on WESAD S2.

Run:    python test_features.py

Expected output:
    - X shape (~101 windows, 10 features)
    - y shape matching, with labels 0-4 represented
    - Per-class feature means table (sanity check: stress should differ
      from baseline in HR, RMSSD, SCR_COUNT — that's the WESAD finding)
    - One PNG plot showing feature trajectories with label backgrounds
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data_loaders import WESADLoader
from features import (extract_wesad_features, FEATURE_NAMES, N_FEATURES)


def main() -> None:
    loader = WESADLoader()
    sid = "S2"
    print(f"Loading {sid} ...")
    d = loader.load(sid)

    print(f"Extracting features (60 s windows) ...")
    X, y = extract_wesad_features(d["signals"], d["fs"], d["labels"])
    print(f"  X shape: {X.shape}    y shape: {y.shape}")
    print(f"  features: {FEATURE_NAMES}")

    # NaN audit
    nan_per_feat = np.isnan(X).sum(axis=0)
    print("\n  NaN counts per feature:")
    for name, n in zip(FEATURE_NAMES, nan_per_feat):
        flag = "" if n == 0 else "  <-- check"
        print(f"    {name:18s}: {n:>4d}{flag}")

    # Per-class summary table
    print("\n  Per-class feature means (labels of interest):")
    print(f"  {'label':<12s}" + "".join(f"{n:>13s}" for n in FEATURE_NAMES))
    for lab in sorted(np.unique(y)):
        lab_name = config.WESAD_LABEL_MAP.get(int(lab), str(lab))
        mask = (y == lab)
        if mask.sum() < 2:
            continue
        means = np.nanmean(X[mask], axis=0)
        row = f"  {lab_name:<12s}" + "".join(f"{m:>13.3f}" for m in means)
        print(row)

    # Plot feature trajectories with label backgrounds
    fig, axes = plt.subplots(N_FEATURES, 1, figsize=(12, 14), sharex=True)
    time_min = np.arange(X.shape[0])  # one tick per window = 1 min

    # Color per label
    label_colors = {
        0: "#dddddd",   # transient
        1: "#cce5ff",   # baseline (blue)
        2: "#ffcccc",   # stress (red)
        3: "#ffe5cc",   # amusement (orange)
        4: "#d4edda",   # meditation (green)
    }

    for ax in axes:
        for i in range(len(y)):
            color = label_colors.get(int(y[i]), "white")
            ax.axvspan(i, i + 1, color=color, alpha=0.6, lw=0)

    for k, name in enumerate(FEATURE_NAMES):
        axes[k].plot(time_min, X[:, k], color="black", lw=0.8)
        axes[k].set_ylabel(name, fontsize=8)
        axes[k].tick_params(labelsize=7)

    axes[-1].set_xlabel("Window (1 min each)")
    axes[0].set_title(f"WESAD {sid} - per-minute features "
                      f"(bg: blue=baseline, red=stress, "
                      f"orange=amusement, green=meditation)")
    plt.tight_layout()
    out = config.REPORTS_DIR / f"features_wesad_{sid}.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\n  Plot saved -> {out}")


if __name__ == "__main__":
    main()