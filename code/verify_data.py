"""
Smoke test for the three data loaders.

Run:    python verify_data.py
Expect:
    - Subject counts and signal shapes printed for each dataset.
    - Three PNG plots saved to ../reports/ for visual sanity-check.
"""
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: write PNGs without opening windows
import matplotlib.pyplot as plt

from data_loaders import WESADLoader, DepresjonLoader, PsykoseLoader
import config


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
def check_wesad() -> None:
    banner("WESAD")
    loader = WESADLoader()
    print(f"Subjects discovered: {len(loader.subjects)}")
    print(f"  -> {loader.subjects}")

    sid = loader.subjects[0]  # S2
    print(f"\nLoading {sid} ...")
    d = loader.load(sid)

    for name, arr in d["signals"].items():
        fs = d["fs"][name]
        dur_min = arr.shape[0] / fs / 60
        shape_str = str(arr.shape)
        print(f"  {name:5s}: shape={shape_str:12s} fs={fs:>3} Hz  duration={dur_min:6.1f} min")

    # Label distribution computed on the BVP-aligned label vector
    bvp_labels = d["labels"]["BVP"]
    unique, counts = np.unique(bvp_labels, return_counts=True)
    print("\n  Label distribution (BVP-aligned):")
    for u, c in zip(unique, counts):
        name = config.WESAD_LABEL_MAP.get(int(u), f"unknown_{u}")
        pct = 100 * c / bvp_labels.size
        flag = "  <-- of interest" if int(u) in config.WESAD_LABELS_OF_INTEREST else ""
        print(f"    {int(u)} ({name:12s}): {c:>10,d} samples ({pct:5.1f}%){flag}")

    # Visual sanity-check plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=False)

    t_bvp = np.arange(d["signals"]["BVP"].shape[0]) / d["fs"]["BVP"] / 60
    axes[0].plot(t_bvp, d["signals"]["BVP"], lw=0.3)
    axes[0].set_ylabel("BVP")
    axes[0].set_title(f"WESAD {sid} - wrist signals")

    t_eda = np.arange(d["signals"]["EDA"].shape[0]) / d["fs"]["EDA"] / 60
    axes[1].plot(t_eda, d["signals"]["EDA"], lw=0.5)
    axes[1].set_ylabel("EDA (uS)")

    t_acc = np.arange(d["signals"]["ACC"].shape[0]) / d["fs"]["ACC"] / 60
    enmo = np.clip(np.linalg.norm(d["signals"]["ACC"], axis=1) - 1.0, 0, None)
    axes[2].plot(t_acc, enmo, lw=0.3)
    axes[2].set_ylabel("ENMO")
    axes[2].set_xlabel("Time (min)")

    plt.tight_layout()
    out = config.REPORTS_DIR / f"verify_wesad_{sid}.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\n  Plot saved -> {out}")


# ----------------------------------------------------------------------------
def check_depresjon() -> None:
    banner("DEPRESJON")
    loader = DepresjonLoader()
    print(f"Condition subjects: {len(loader.condition_subjects)}")
    print(f"Control subjects:   {len(loader.control_subjects)}")
    print(f"scores.csv present: {loader._scores is not None}")
    if loader._scores is not None:
        print(f"scores.csv columns: {list(loader._scores.columns)}")

    sid = "condition_1"
    print(f"\nLoading {sid} ...")
    d = loader.load(sid)
    act = d["signals"]["ACTIVITY"]
    print(f"  ACTIVITY shape: {act.shape}  duration ~ {act.shape[0]/60/24:.2f} days")
    print(f"  stats: mean={act.mean():.1f}  max={act.max():.0f}  "
          f"n_zero={(act == 0).sum():,d}  ({100*(act == 0).mean():.1f}% zeros)")
    print(f"  subject label: {d['subject_label']} (1=condition, 0=control)")
    print(f"  meta keys: {list(d['meta'].keys())}")

    d_ctrl = loader.load("control_1")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].plot(act, lw=0.3, color="firebrick")
    axes[0].set_title(f"DEPRESJON condition_1 (label={d['subject_label']})")
    axes[0].set_ylabel("Activity")
    axes[1].plot(d_ctrl["signals"]["ACTIVITY"], lw=0.3, color="steelblue")
    axes[1].set_title(f"DEPRESJON control_1 (label={d_ctrl['subject_label']})")
    axes[1].set_ylabel("Activity")
    axes[1].set_xlabel("Minutes from start of recording")
    plt.tight_layout()
    out = config.REPORTS_DIR / "verify_depresjon.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\n  Plot saved -> {out}")


# ----------------------------------------------------------------------------
def check_psykose() -> None:
    banner("PSYKOSE")
    loader = PsykoseLoader()
    print(f"Patient subjects: {len(loader.patient_subjects)}")
    print(f"Control subjects: {len(loader.control_subjects)}")
    print(f"patients_info.csv present: {loader._info is not None}")
    if loader._info is not None:
        print(f"patients_info.csv columns: {list(loader._info.columns)}")
    print(f"days.csv present: {loader._days is not None}")

    sid = "patient_1"
    print(f"\nLoading {sid} ...")
    d = loader.load(sid)
    act = d["signals"]["ACTIVITY"]
    print(f"  ACTIVITY shape: {act.shape}  duration ~ {act.shape[0]/60/24:.2f} days")
    print(f"  stats: mean={act.mean():.1f}  max={act.max():.0f}  "
          f"n_zero={(act == 0).sum():,d}  ({100*(act == 0).mean():.1f}% zeros)")
    print(f"  subject label: {d['subject_label']} (1=patient, 0=control)")

    d_ctrl = loader.load("control_1")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].plot(act, lw=0.3, color="darkorange")
    axes[0].set_title(f"PSYKOSE patient_1 (label={d['subject_label']})")
    axes[0].set_ylabel("Activity")
    axes[1].plot(d_ctrl["signals"]["ACTIVITY"], lw=0.3, color="seagreen")
    axes[1].set_title(f"PSYKOSE control_1 (label={d_ctrl['subject_label']})")
    axes[1].set_ylabel("Activity")
    axes[1].set_xlabel("Minutes from start of recording")
    plt.tight_layout()
    out = config.REPORTS_DIR / "verify_psykose.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\n  Plot saved -> {out}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    failures = []
    for name, fn in [("WESAD", check_wesad),
                     ("DEPRESJON", check_depresjon),
                     ("PSYKOSE", check_psykose)]:
        try:
            fn()
        except Exception as e:
            print(f"\n{name} check FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("All three datasets loaded successfully.")
    print(f"Verification plots: {config.REPORTS_DIR}")
    print("=" * 70)