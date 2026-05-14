"""
Extract features from DEPRESJON and PSYKOSE (activity-only datasets).

Both datasets provide per-minute wrist actigraphy counts. We compute
features over non-overlapping 60-minute windows to match the WESAD
feature time scale.

Features per 60-minute window (8 features):
    0  ACT_mean        mean activity count
    1  ACT_std         std of activity counts
    2  ACT_zero_frac   fraction of zero-count minutes (inactivity)
    3  ACT_p90         90th percentile count (peak activity bursts)
    4  M10             mean of the 10 most active minutes (circadian vigour)
    5  L5              mean of the 5 least active minutes (rest depth)
    6  RA              relative amplitude = (M10 - L5) / (M10 + L5)
    7  IV              intradaily variability (fragmentation index)

M10, L5, RA, IV are established non-parametric circadian rhythm metrics
(van Someren et al.; widely used in actigraphy literature).

Output: features/activity_features.npz
    X           (N, 8)  float32   feature matrix
    y           (N,)    int8      0=control, 1=depression, 2=schizophrenia
    subject     (N,)    <U16      subject id
    dataset     (N,)    <U10      'depresjon' or 'psykose'
    window_idx  (N,)    int32     window index within subject
"""
from __future__ import annotations
import numpy as np
from tqdm import tqdm

import config
from data_loaders import DepresjonLoader, PsykoseLoader

WINDOW_MIN = 60          # 60-minute windows
FEATURE_NAMES = (
    "ACT_mean", "ACT_std", "ACT_zero_frac", "ACT_p90",
    "M10", "L5", "RA", "IV",
)
N_FEATURES = len(FEATURE_NAMES)


def extract_circadian_metrics(act: np.ndarray) -> tuple[float, float, float, float]:
    """
    M10, L5, RA, IV from a multi-day activity array (1-min samples).

    M10: mean of the 10 highest-count minutes across the recording.
    L5:  mean of the 5  lowest-count  minutes across the recording.
    RA:  (M10 - L5) / (M10 + L5)  — 0 = no rhythm, 1 = perfect rhythm.
    IV:  intradaily variability = mean squared successive difference /
         overall variance.  High IV = fragmented, unstable rhythm.
    """
    if act.size < 15:
        return (float("nan"),) * 4

    sorted_act = np.sort(act.astype(np.float32))
    m10 = float(sorted_act[-10:].mean())
    l5  = float(sorted_act[:5].mean())
    denom = m10 + l5
    ra  = float((m10 - l5) / denom) if denom > 0 else float("nan")

    if act.size < 3:
        iv = float("nan")
    else:
        num = float(np.mean(np.diff(act.astype(np.float32)) ** 2))
        var = float(act.astype(np.float32).var(ddof=0))
        iv  = (num / var) if var > 0 else float("nan")

    return m10, l5, ra, iv


def extract_window(seg: np.ndarray) -> np.ndarray:
    """Compute 8-feature vector for a 60-minute activity window."""
    out = np.full(N_FEATURES, np.nan, dtype=np.float32)
    if seg.size == 0:
        return out
    x = seg.astype(np.float32)
    out[0] = float(x.mean())
    out[1] = float(x.std(ddof=0))
    out[2] = float((x == 0).mean())
    out[3] = float(np.percentile(x, 90))
    m10, l5, ra, iv = extract_circadian_metrics(x)
    out[4] = m10
    out[5] = l5
    out[6] = ra
    out[7] = iv
    return out


def extract_subject(activity: np.ndarray,
                    subject_label: int,
                    subject_id: str,
                    dataset_name: str,
                    window_min: int = WINDOW_MIN
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_windows = activity.size // window_min
    if n_windows == 0:
        empty = np.empty((0, N_FEATURES), dtype=np.float32)
        return (empty,
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype="<U16"),
                np.empty(0, dtype="<U10"))

    X = np.empty((n_windows, N_FEATURES), dtype=np.float32)
    for i in range(n_windows):
        seg = activity[i * window_min:(i + 1) * window_min]
        X[i] = extract_window(seg)

    y       = np.full(n_windows, subject_label, dtype=np.int8)
    subj    = np.full(n_windows, subject_id,    dtype="<U16")
    dset    = np.full(n_windows, dataset_name,  dtype="<U10")
    return X, y, subj, dset


def main():
    all_X, all_y, all_subj, all_dset, all_widx = [], [], [], [], []

    # Label scheme: 0=control, 1=depression, 2=schizophrenia
    tasks = [
        (DepresjonLoader(), "depresjon", {"condition": 1, "control": 0}),
        (PsykoseLoader(),   "psykose",   {"patient":    2, "control": 0}),
    ]

    for loader, dset_name, label_map in tasks:
        print(f"\n{dset_name.upper()}")
        rows = []
        for sid in tqdm(loader.subjects, desc=f"  {dset_name}"):
            d = loader.load(sid)
            activity = d["signals"]["ACTIVITY"]
            prefix   = sid.split("_")[0]        # 'condition', 'control', or 'patient'
            label    = label_map[prefix]

            X, y, subj, dset = extract_subject(
                activity, label, sid, dset_name
            )
            n_w = X.shape[0]
            rows.append((sid, label, activity.size, n_w))

            if n_w > 0:
                widx = np.arange(n_w, dtype=np.int32)
                all_X.append(X); all_y.append(y)
                all_subj.append(subj); all_dset.append(dset)
                all_widx.append(widx)

        print(f"  {'subject':<18s} {'label':>5} {'mins':>6} {'windows':>8}")
        print("  " + "-" * 42)
        for sid, lab, mins, nw in rows:
            print(f"  {sid:<18s} {lab:>5d} {mins:>6d} {nw:>8d}")

    X_all    = np.concatenate(all_X,    axis=0)
    y_all    = np.concatenate(all_y,    axis=0)
    subj_all = np.concatenate(all_subj, axis=0)
    dset_all = np.concatenate(all_dset, axis=0)
    widx_all = np.concatenate(all_widx, axis=0).astype(np.int32)

    out = config.FEATURES_DIR / "activity_features.npz"
    np.savez_compressed(
        out,
        X=X_all, y=y_all, subject=subj_all,
        dataset=dset_all, window_idx=widx_all,
        feature_names=np.array(FEATURE_NAMES, dtype="<U16"),
    )

    print(f"\nSaved: {out}  ({out.stat().st_size/1024:.1f} KB)")
    print(f"Shape: X={X_all.shape}, y={y_all.shape}")

    label_names = {0: "control", 1: "depression", 2: "schizophrenia"}
    print("\nClass distribution:")
    for lab, name in label_names.items():
        c = int((y_all == lab).sum())
        print(f"  {lab} ({name:14s}): {c:>5d}  ({100*c/y_all.size:5.1f}%)")

    print("\nNaN counts per feature:")
    for i, name in enumerate(FEATURE_NAMES):
        n = int(np.isnan(X_all[:, i]).sum())
        pct = 100 * n / X_all.shape[0]
        flag = "  <-- check" if pct > 5 else ""
        print(f"  {name:<16s}: {n:>4d}  ({pct:4.1f}%){flag}")

    print("\nPer-class feature means:")
    print(f"  {'label':<15s}" + "".join(f"{n:>13s}" for n in FEATURE_NAMES))
    for lab, name in label_names.items():
        mask = (y_all == lab)
        if not mask.any():
            continue
        means = np.nanmean(X_all[mask], axis=0)
        row = f"  {name:<15s}" + "".join(f"{m:>13.3f}" for m in means)
        print(row)


if __name__ == "__main__":
    main()