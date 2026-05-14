"""
Load the saved wesad_features.npz and produce a per-feature density plot,
split by class. Lets us visually confirm that features separate the classes.

Run:    python inspect_wesad_features.py
Output: reports/wesad_features_by_class.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
LABEL_COLORS = {1: "steelblue", 2: "firebrick", 3: "darkorange", 4: "seagreen"}


def main():
    path = config.FEATURES_DIR / "wesad_features.npz"
    print(f"Loading {path} ...")
    z = np.load(path, allow_pickle=False)
    X = z["X"]
    y = z["y"]
    subjects = z["subject"]
    feature_names = list(z["feature_names"])

    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  unique subjects: {len(np.unique(subjects))}")
    print(f"  feature names: {feature_names}")

    n_feat = len(feature_names)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()

    for k, name in enumerate(feature_names):
        ax = axes[k]
        col = X[:, k]
        # Decide range from finite values across all classes
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            ax.set_title(f"{name}\n(all NaN)")
            continue
        lo, hi = np.percentile(finite, [1, 99])
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, 40)

        for lab in [1, 2, 3, 4]:
            vals = col[(y == lab) & np.isfinite(col)]
            if vals.size == 0:
                continue
            ax.hist(vals, bins=bins, density=True, alpha=0.45,
                    color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("WESAD features by class (all 15 subjects, n=741 windows)",
                 fontsize=12)
    plt.tight_layout()
    out = config.REPORTS_DIR / "wesad_features_by_class.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nPlot saved -> {out}")

    # Quick text-based separability check: for each feature, compute
    # Cohen's d between baseline and stress (NaN-ignoring)
    print("\nFeature separability (Cohen's d, baseline vs stress):")
    print(f"  {'feature':<18s} {'|d|':>6s}  interpretation")
    print("  " + "-" * 50)
    for k, name in enumerate(feature_names):
        b = X[(y == 1), k]
        s = X[(y == 2), k]
        b = b[np.isfinite(b)]
        s = s[np.isfinite(s)]
        if b.size < 5 or s.size < 5:
            print(f"  {name:<18s}   ---  insufficient data")
            continue
        mean_b, mean_s = b.mean(), s.mean()
        # pooled std
        var_b, var_s = b.var(ddof=1), s.var(ddof=1)
        pooled = np.sqrt((var_b + var_s) / 2)
        if pooled == 0:
            print(f"  {name:<18s}   ---  zero variance")
            continue
        d = (mean_s - mean_b) / pooled
        absd = abs(d)
        if absd < 0.2:
            tag = "negligible"
        elif absd < 0.5:
            tag = "small"
        elif absd < 0.8:
            tag = "medium"
        else:
            tag = "LARGE"
        sign = "+" if d > 0 else "-"
        print(f"  {name:<18s}  {sign}{absd:5.2f}  {tag}")


if __name__ == "__main__":
    main()