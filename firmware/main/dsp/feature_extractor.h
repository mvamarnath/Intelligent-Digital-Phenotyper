#pragma once
#include <stdint.h>
#include "iir_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Feature extractor — C port of features.py v1.
 *
 * Computes one 10-element feature vector per 60-second window.
 * Feature order is fixed — matches the trained TFLM models exactly.
 *
 * Index  Name             Sensor   Units
 * -----  ---------------  -------  ------
 *   0    ENMO_mean        ACC      g
 *   1    ENMO_std         ACC      g
 *   2    DOM_FREQ         ACC      Hz  (NaN if ENMO_mean < 0.02)
 *   3    DOM_FREQ_POWER   ACC      0-1 (NaN if ENMO_mean < 0.02)
 *   4    HR_MEAN          BVP      BPM
 *   5    RMSSD            BVP      ms
 *   6    SDNN             BVP      ms
 *   7    SCL              EDA      uS
 *   8    SCR_COUNT        EDA      count/min
 *   9    TEMP_MEAN        TEMP     degC
 *
 * NaN is represented as IEEE 754 NaN (nanf("")).
 * The TFLM preprocessing (imputer + scaler) handles NaN before inference.
 */

#define FE_N_FEATURES       10
#define FE_WINDOW_S         60
#define FE_ACC_FS           32
#define FE_BVP_FS           64
#define FE_EDA_FS            4
#define FE_TEMP_FS           4

/* Samples per 60-second window at each rate */
#define FE_ACC_WINDOW_N     (FE_WINDOW_S * FE_ACC_FS)   /* 1920 */
#define FE_BVP_WINDOW_N     (FE_WINDOW_S * FE_BVP_FS)   /* 3840 */
#define FE_EDA_WINDOW_N     (FE_WINDOW_S * FE_EDA_FS)   /*  240 */

/* FFT size for dominant frequency (4 s at 32 Hz) */
#define FE_FFT_N            128
#define FE_FFT_WORK_N       (FE_FFT_N * 2)

/* Motion gate for HRV — skip cardiac features if ENMO std > threshold */
#define FE_MOTION_GATE_G    0.05f

/* ENMO threshold below which DOM_FREQ is set to NaN */
#define FE_STILL_THRESH_G   0.02f

/* BVP peak detector constants (Elgendi algorithm) */
#define FE_BVP_W_PEAK_S     0.111f   /* short MA window ~111 ms */
#define FE_BVP_W_BEAT_S     0.667f   /* long  MA window ~667 ms */
#define FE_BVP_REFRAC_S     0.300f   /* minimum inter-peak interval */
#define FE_BVP_BAND_LO      0.5f
#define FE_BVP_BAND_HI      8.0f

/* IBI plausibility limits */
#define FE_IBI_MIN_MS       300.0f
#define FE_IBI_MAX_MS       2000.0f
#define FE_IBI_OUTLIER_PCT  0.15f    /* Berntson 15% local-median threshold */
#define FE_IBI_MIN_CLEAN    5        /* minimum IBIs after cleaning */

/* EDA */
#define FE_EDA_TONIC_CUTOFF 0.05f    /* low-pass for SCL */
#define FE_EDA_SCR_THRESH   0.05f    /* minimum phasic amplitude for SCR */

/**
 * Persistent filter states — kept across windows so the IIR filters
 * do not reset at every 60-second boundary (matches Python behaviour
 * where filters run on the full recording).
 */
typedef struct {
    IIR1State eda_tonic_lp;   /* EDA tonic low-pass state */
} FEFilterState;

/**
 * Initialise filter states to zero. Call once at startup.
 */
void fe_init(FEFilterState* fs);

/**
 * Extract one 10-element feature vector from a 60-second window.
 *
 * acc:   (FE_ACC_WINDOW_N, 3) float32, units = g (already /64 if E4)
 * bvp:   (FE_BVP_WINDOW_N,)  float32, raw BVP counts
 * eda:   (FE_EDA_WINDOW_N,)  float32, uS
 * temp:  (FE_EDA_WINDOW_N,)  float32, degC
 * fs:    persistent filter state
 * out:   caller-provided float32[FE_N_FEATURES]
 *
 * NaN entries in out indicate features that could not be computed
 * (motion gate, insufficient peaks, zero-variance signal).
 */
void fe_extract_window(
    const float* acc,   /* (N*3) row-major: x0,y0,z0, x1,y1,z1, ... */
    const float* bvp,
    const float* eda,
    const float* temp,
    FEFilterState* fs,
    float* out
);

/* ── Individual feature functions (exposed for unit testing) ─────────────── */

/**
 * Compute ENMO for each sample.
 * acc_xyz: (n*3) float32 row-major
 * enmo:    (n,)  float32 output — clipped to [0, inf)
 */
void fe_enmo(const float* acc_xyz, float* enmo, int n);

/**
 * Compute mean HR, RMSSD, SDNN from BVP window.
 * Returns false if insufficient clean IBIs.
 * acc_enmo: same-window ENMO for motion gate (may be NULL to skip gate)
 */
bool fe_cardiac(const float* bvp, int bvp_n, int bvp_fs,
                const float* acc_enmo, int acc_n,
                float* hr_bpm, float* rmssd_ms, float* sdnn_ms);

/**
 * Compute SCL and SCR count from EDA window.
 * fs: persistent filter state (tonic LP carries over across windows)
 */
void fe_eda(const float* eda, int n, int eda_fs,
            FEFilterState* fs,
            float* scl_us, int* scr_count);

#ifdef __cplusplus
}
#endif