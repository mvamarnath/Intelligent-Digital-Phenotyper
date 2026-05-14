"""
Head C: Autonomic Autoencoder — trains on WESAD baseline windows only.

Architecture: 1D convolutional autoencoder (input -> latent -> reconstruction).
Training strategy: Leave-One-Subject-Out (LOSO) cross-validation.
Each fold trains on 14 subjects' baseline windows, evaluates reconstruction
error on all classes of the held-out subject.

The anomaly score is reconstruction error (MSE). A window is flagged as
anomalous when its error exceeds the 95th percentile of the subject's own
baseline errors — this threshold is computed per-subject at deploy time.

Output:
    models/autoencoder_fold_{sid}.keras   — one model per LOSO fold
    models/autoencoder_final.keras        — trained on ALL subjects (deploy model)
    models/autoencoder_thresholds.npz     — per-subject 95th-pct thresholds
    reports/autoencoder_loso_results.txt  — AUROC per fold
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF info spam

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

import config

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM   = 4      # bottleneck size — keeps model tiny for TFLM
EPOCHS       = 80
BATCH_SIZE   = 32
LR           = 1e-3
ANOMALY_PCT  = 95     # threshold percentile on baseline reconstruction errors

BASELINE_LABEL = 1    # WESAD label 1 = baseline


# ── Model definition ─────────────────────────────────────────────────────────
def build_autoencoder(n_features: int) -> keras.Model:
    """
    Symmetric MLP autoencoder.
    Input/output: (n_features,) float32.

    We use a simple MLP rather than 1D CNN because our feature vector is
    10-dimensional (not a raw waveform). CNN spatial priors don't apply here;
    MLP is smaller and faster to quantise to INT8 for TFLM.

    Architecture:
        Encoder: 10 -> 8 -> LATENT_DIM
        Decoder: LATENT_DIM -> 8 -> 10
    Total parameters: ~330  (trivially fits in TFLM on ESP32-S3)
    """
    inp = keras.Input(shape=(n_features,), name="input")

    # Encoder
    x = keras.layers.Dense(8, activation="relu", name="enc_1")(inp)
    z = keras.layers.Dense(LATENT_DIM, activation="relu", name="latent")(x)

    # Decoder
    x = keras.layers.Dense(8, activation="relu", name="dec_1")(z)
    out = keras.layers.Dense(n_features, activation="linear", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="mse",
    )
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(X_train: np.ndarray,
               X_test:  np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, SimpleImputer, RobustScaler]:
    """
    1. Impute NaNs with per-feature median (computed on train only).
    2. RobustScaler (median/IQR) — resistant to outliers in EDA/RMSSD.
    Both fitted on training data only; applied to test without leakage.
    """
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i  = imputer.transform(X_test)

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train_i).astype(np.float32)
    X_test_s  = scaler.transform(X_test_i).astype(np.float32)

    return X_train_s, X_test_s, imputer, scaler


# ── LOSO cross-validation ─────────────────────────────────────────────────────
def run_loso(X: np.ndarray, y: np.ndarray, subjects: np.ndarray):
    unique_subjects = list(config.WESAD_SUBJECTS)
    n_feat = X.shape[1]

    fold_results = []
    all_thresholds = {}

    print(f"\n{'fold':<6} {'subject':<8} {'n_train_base':>12} "
          f"{'n_test_base':>11} {'AUROC (base vs stress)':>22}")
    print("-" * 65)

    for sid in unique_subjects:
        test_mask  = (subjects == sid)
        train_mask = ~test_mask

        X_train_all = X[train_mask]
        y_train_all = y[train_mask]
        X_test      = X[test_mask]
        y_test      = y[test_mask]

        # Train autoencoder ONLY on baseline windows from training subjects
        train_base_mask = (y_train_all == BASELINE_LABEL)
        X_train_base = X_train_all[train_base_mask]

        if X_train_base.shape[0] < 10:
            print(f"  {sid}: insufficient baseline windows, skipping.")
            continue

        X_train_s, X_test_s, imputer, scaler = preprocess(
            X_train_base, X_test
        )

        model = build_autoencoder(n_feat)
        model.fit(
            X_train_s, X_train_s,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10,
                    restore_best_weights=True, verbose=0
                )
            ]
        )

        # Reconstruction error on test subject
        X_test_recon = model.predict(X_test_s, verbose=0)
        errors = np.mean((X_test_s - X_test_recon) ** 2, axis=1)

        # Per-subject anomaly threshold = 95th pct of baseline errors
        base_mask_test = (y_test == BASELINE_LABEL)
        if base_mask_test.sum() > 0:
            base_errors = errors[base_mask_test]
            threshold = float(np.percentile(base_errors, ANOMALY_PCT))
        else:
            threshold = float(np.percentile(errors, ANOMALY_PCT))
        all_thresholds[sid] = threshold

        # AUROC: baseline (0) vs stress (1) — our primary clinical signal
        stress_mask = (y_test == 2)   # label 2 = stress
        eval_mask = base_mask_test | stress_mask
        if eval_mask.sum() > 1 and stress_mask.sum() > 0:
            y_bin = (y_test[eval_mask] == 2).astype(int)
            e_eval = errors[eval_mask]
            auroc = roc_auc_score(y_bin, e_eval)
        else:
            auroc = float("nan")

        n_train_base = X_train_base.shape[0]
        n_test_base  = int(base_mask_test.sum())
        print(f"  {sid:<8} {n_train_base:>12d} {n_test_base:>11d} "
              f"{auroc:>22.3f}")

        fold_results.append({
            "subject": sid, "auroc": auroc, "threshold": threshold,
            "model": model, "imputer": imputer, "scaler": scaler,
        })

        # Save per-fold model
        model.save(config.MODELS_DIR / f"autoencoder_fold_{sid}.keras")

    return fold_results, all_thresholds


# ── Final model (all subjects) ────────────────────────────────────────────────
def train_final(X: np.ndarray, y: np.ndarray):
    """Train on ALL subjects' baseline windows — this is the deploy model."""
    base_mask = (y == BASELINE_LABEL)
    X_base = X[base_mask]

    imputer = SimpleImputer(strategy="median")
    X_i = imputer.fit_transform(X_base)
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X_i).astype(np.float32)

    model = build_autoencoder(X_s.shape[1])
    model.fit(
        X_s, X_s,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10,
                restore_best_weights=True, verbose=0
            )
        ]
    )
    return model, imputer, scaler


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load features
    path = config.FEATURES_DIR / "wesad_features.npz"
    z = np.load(path, allow_pickle=False)
    X, y, subjects = z["X"], z["y"], z["subject"]
    feature_names = list(z["feature_names"])

    print(f"Loaded: X={X.shape}, y={y.shape}, subjects={len(np.unique(subjects))}")
    print(f"Features: {feature_names}")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    # LOSO
    print("\n=== LOSO Cross-Validation ===")
    fold_results, thresholds = run_loso(X, y, subjects)

    # Summary
    aurocs = [r["auroc"] for r in fold_results if not np.isnan(r["auroc"])]
    print(f"\nLOSO AUROC:  mean={np.mean(aurocs):.3f}  "
          f"std={np.std(aurocs):.3f}  "
          f"min={np.min(aurocs):.3f}  "
          f"max={np.max(aurocs):.3f}")

    # Final model
    print("\n=== Training final model (all subjects) ===")
    final_model, final_imputer, final_scaler = train_final(X, y)
    final_model.save(config.MODELS_DIR / "autoencoder_final.keras")
    print(f"Final model saved -> {config.MODELS_DIR / 'autoencoder_final.keras'}")
    final_model.summary()

    # Save thresholds + preprocessing params
    np.savez(
        config.MODELS_DIR / "autoencoder_thresholds.npz",
        subjects=np.array(list(thresholds.keys())),
        thresholds=np.array(list(thresholds.values()), dtype=np.float32),
        imputer_medians=final_imputer.statistics_.astype(np.float32),
        scaler_center=final_scaler.center_.astype(np.float32),
        scaler_scale=final_scaler.scale_.astype(np.float32),
    )
    print(f"Thresholds saved -> {config.MODELS_DIR / 'autoencoder_thresholds.npz'}")

    # Write text report
    report_path = config.REPORTS_DIR / "autoencoder_loso_results.txt"
    with open(report_path, "w") as f:
        f.write("Autoencoder LOSO Results\n")
        f.write("=" * 40 + "\n")
        for r in fold_results:
            f.write(f"{r['subject']}  AUROC={r['auroc']:.3f}  "
                    f"threshold={r['threshold']:.4f}\n")
        f.write(f"\nMean AUROC: {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}\n")
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()