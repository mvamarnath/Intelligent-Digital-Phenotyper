"""
Extract features for all 15 WESAD subjects and save a single .npz file.

Output: features/wesad_features.npz with arrays:
    X         (N, 10) float32  - per-window feature matrix
    y         (N,)    int8     - per-window label (1=base, 2=stress, 3=amuse, 4=med)
    subject   (N,)    <U4      - subject ID per window (e.g. 'S2', 'S15')
    window_idx (N,)   int32    - index of the window within its subject

Run:    python extract_wesad_all.py
"""
from __future__ import annotations
import time
import numpy as np
from tqdm import tqdm

import config
from data_loaders import WESADLoader
from features import extract_wesad_features, FEATURE_NAMES


# Labels we keep. Everything else (0=transient, 5/6/7=ignore) is dropped.
KEEP_LABELS = {1, 2, 3, 4}
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def main():
    loader = WESADLoader()
    n_subjects = len(loader.subjects)
    print(f"Extracting features for {n_subjects} WESAD subjects ...\n")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_subject: list[np.ndarray] = []
    all_window_idx: list[np.ndarray] = []

    t0 = time.time()
    per_subject_summary = []

    for sid in tqdm(loader.subjects, desc="Subjects"):
        d = loader.load(sid)
        X, y = extract_wesad_features(d["signals"], d["fs"], d["labels"])

        # Build a mask for the windows we want to keep
        keep_mask = np.isin(y, list(KEEP_LABELS))
        n_total = y.size
        n_kept  = int(keep_mask.sum())

        if n_kept == 0:
            print(f"  WARNING: {sid} has no windows in KEEP_LABELS; skipping.")
            per_subject_summary.append((sid, n_total, 0, 0, 0, 0, 0))
            continue

        Xk = X[keep_mask]
        yk = y[keep_mask]
        widx = np.flatnonzero(keep_mask).astype(np.int32)
        subj = np.full(yk.shape, sid, dtype="<U4")

        all_X.append(Xk)
        all_y.append(yk)
        all_subject.append(subj)
        all_window_idx.append(widx)

        # Per-class counts for this subject
        counts = {lab: int((yk == lab).sum()) for lab in KEEP_LABELS}
        per_subject_summary.append((
            sid, n_total, n_kept,
            counts[1], counts[2], counts[3], counts[4],
        ))

    X_all       = np.concatenate(all_X, axis=0).astype(np.float32)
    y_all       = np.concatenate(all_y, axis=0).astype(np.int8)
    subject_all = np.concatenate(all_subject, axis=0)
    window_idx  = np.concatenate(all_window_idx, axis=0).astype(np.int32)

    elapsed = time.time() - t0

    # ------- Save -------
    out_path = config.FEATURES_DIR / "wesad_features.npz"
    np.savez_compressed(
        out_path,
        X=X_all, y=y_all, subject=subject_all, window_idx=window_idx,
        feature_names=np.array(FEATURE_NAMES, dtype="<U16"),
    )

    # ------- Summary -------
    print(f"\nDone in {elapsed:.1f} s.")
    print(f"Output: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nShape: X={X_all.shape}, y={y_all.shape}")

    print("\nPer-subject window counts:")
    header = f"  {'sid':<5} {'total':>7} {'kept':>6} " \
             f"{'base':>6} {'stress':>7} {'amuse':>6} {'med':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in per_subject_summary:
        sid, ntot, nkept, nb, ns, na, nm = row
        print(f"  {sid:<5} {ntot:>7d} {nkept:>6d} "
              f"{nb:>6d} {ns:>7d} {na:>6d} {nm:>5d}")

    print("\nOverall class distribution (kept windows):")
    for lab in sorted(KEEP_LABELS):
        c = int((y_all == lab).sum())
        pct = 100 * c / y_all.size
        print(f"  {lab} ({LABEL_NAMES[lab]:10s}): {c:>5d}  ({pct:5.1f}%)")

    # NaN audit
    print("\nNaN counts per feature (across all kept windows):")
    for i, name in enumerate(FEATURE_NAMES):
        n_nan = int(np.isnan(X_all[:, i]).sum())
        pct = 100 * n_nan / X_all.shape[0]
        flag = "" if pct < 5 else "  <-- elevated"
        print(f"  {name:18s}: {n_nan:>5d}  ({pct:5.1f}%){flag}")

    # Per-class feature means (sanity check)
    print("\nPer-class feature means (across all subjects, NaN-ignoring):")
    print(f"  {'label':<12s}" + "".join(f"{n:>13s}" for n in FEATURE_NAMES))
    for lab in sorted(KEEP_LABELS):
        mask = (y_all == lab)
        if not mask.any():
            continue
        with np.errstate(invalid="ignore"):
            means = np.nanmean(X_all[mask], axis=0)
        row = f"  {LABEL_NAMES[lab]:<12s}" + "".join(f"{m:>13.3f}" for m in means)
        print(row)


if __name__ == "__main__":
    main()