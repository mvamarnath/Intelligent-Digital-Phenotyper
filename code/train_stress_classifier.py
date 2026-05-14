"""
Head D: Stress/Affect Classifier
3-class MLP: baseline(0) vs stress(1) vs other(2: amusement+meditation)
LOSO cross-validation. Exports stress_clf_int8.tflite.
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import config

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS     = 100
BATCH      = 32
LR         = 1e-3

# Remap WESAD labels -> 3 classes
# 1=baseline->0, 2=stress->1, 3=amusement->2, 4=meditation->2
LABEL_MAP  = {1: 0, 2: 1, 3: 2, 4: 2}
CLASS_NAMES = ["baseline", "stress", "other"]


def remap(y):
    return np.array([LABEL_MAP[int(l)] for l in y], dtype=np.int8)


def preprocess(X_tr, X_te):
    imp = SimpleImputer(strategy="median")
    sc  = RobustScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    Xte = sc.transform(imp.transform(X_te)).astype(np.float32)
    return Xtr, Xte, imp, sc


def build_model(n_feat):
    inp = keras.Input(shape=(n_feat,))
    x   = keras.layers.Dense(16, activation="relu")(inp)
    x   = keras.layers.Dropout(0.2)(x)
    x   = keras.layers.Dense(8,  activation="relu")(x)
    out = keras.layers.Dense(3,  activation="softmax")(x)
    m   = keras.Model(inp, out, name="stress_clf")
    m.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def main():
    z = np.load(config.FEATURES_DIR / "wesad_features.npz", allow_pickle=False)
    X, y_raw, subjects = z["X"], z["y"], z["subject"]
    y = remap(y_raw)

    print(f"Loaded X={X.shape}  classes={dict(zip(*np.unique(y, return_counts=True)))}")

    # ── LOSO ──────────────────────────────────────────────────────────────────
    print(f"\n{'subject':<8} {'acc':>6}  per-class (base / stress / other)")
    print("-" * 55)

    all_true, all_pred = [], []
    for sid in config.WESAD_SUBJECTS:
        te  = subjects == sid
        tr  = ~te
        Xtr, Xte, imp, sc = preprocess(X[tr], X[te])
        ytr, yte = y[tr], y[te]

        # Class weights to handle imbalance
        counts  = np.bincount(ytr, minlength=3).astype(float)
        weights = {i: ytr.size / (3 * max(c, 1)) for i, c in enumerate(counts)}

        m = build_model(Xtr.shape[1])
        m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH,
              class_weight=weights, validation_split=0.1, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=12,
                  restore_best_weights=True, verbose=0)])

        pred = np.argmax(m.predict(Xte, verbose=0), axis=1)
        acc  = float((pred == yte).mean())
        all_true.extend(yte); all_pred.extend(pred)

        per = []
        for c in range(3):
            mask = yte == c
            per.append(f"{float((pred[mask]==c).mean()):.2f}" if mask.any() else " -- ")
        print(f"  {sid:<6} {acc:>6.3f}  {' / '.join(per)}")

    print(f"\nOverall accuracy: {float(np.mean(np.array(all_true)==np.array(all_pred))):.3f}")
    print(classification_report(all_true, all_pred,
                                target_names=CLASS_NAMES, digits=3))

    # ── Final model (all subjects) ─────────────────────────────────────────────
    imp_f = SimpleImputer(strategy="median")
    sc_f  = RobustScaler()
    Xs    = sc_f.fit_transform(imp_f.fit_transform(X)).astype(np.float32)
    ys    = y

    mf = build_model(Xs.shape[1])
    mf.fit(Xs, ys, epochs=EPOCHS, batch_size=BATCH,
           validation_split=0.1, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(
               monitor="val_loss", patience=12,
               restore_best_weights=True, verbose=0)])
    mf.save(config.MODELS_DIR / "stress_clf_final.keras")
    mf.summary()

    # ── TFLite INT8 export ─────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(mf)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    rep_data = Xs[ys == 0]   # baseline as representative
    def rep_gen():
        for s in rep_data:
            yield [s.reshape(1, -1)]
    converter.representative_dataset = rep_gen

    tfl = converter.convert()
    out = config.MODELS_DIR / "stress_clf_int8.tflite"
    out.write_bytes(tfl)
    print(f"\nTFLite INT8: {out}  ({len(tfl):,d} bytes / {len(tfl)/1024:.1f} KB)")

    # Save preprocessing params as C header
    hdr = config.MODELS_DIR / "stress_clf_quant_params.h"
    interp = tf.lite.Interpreter(model_path=str(out))
    interp.allocate_tensors()
    si, zi = interp.get_input_details()[0]["quantization"]
    so, zo = interp.get_output_details()[0]["quantization"]
    with open(hdr, "w") as f:
        f.write("// Auto-generated by train_stress_classifier.py\n#pragma once\n\n")
        f.write(f"#define SC_N_FEATURES  {Xs.shape[1]}\n")
        f.write(f"#define SC_N_CLASSES   3\n\n")
        f.write(f"static const float SC_IMPUTER_MEDIAN[{Xs.shape[1]}] = "
                f"{{{', '.join(f'{v:.6f}f' for v in imp_f.statistics_)}}};\n")
        f.write(f"static const float SC_SCALER_CENTER[{Xs.shape[1]}] = "
                f"{{{', '.join(f'{v:.6f}f' for v in sc_f.center_)}}};\n")
        f.write(f"static const float SC_SCALER_SCALE[{Xs.shape[1]}] = "
                f"{{{', '.join(f'{v:.6f}f' for v in sc_f.scale_)}}};\n\n")
        f.write(f"#define SC_INPUT_SCALE  {si:.8f}f\n#define SC_INPUT_ZP  {zi}\n")
        f.write(f"#define SC_OUTPUT_SCALE {so:.8f}f\n#define SC_OUTPUT_ZP {zo}\n")
    print(f"C header: {hdr}")


if __name__ == "__main__":
    main()