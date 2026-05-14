#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Head C: Autonomic Autoencoder
 *
 * Loads autoencoder_int8.tflite from the models SPIFFS partition.
 * Preprocesses the 10-feature vector (impute NaN + RobustScale),
 * runs INT8 inference, returns reconstruction MSE as anomaly score.
 *
 * Anomaly score > threshold -> flag as anomalous window.
 * Threshold is loaded from NVS (set during personal calibration).
 */

#define HC_N_FEATURES   10
#define HC_MODEL_PATH   "/models/autoencoder_int8.tflite"
#define HC_NVS_NS       "head_c"
#define HC_NVS_THRESH   "threshold"
#define HC_DEFAULT_THRESH 2.0f   /* conservative default before calibration */

typedef struct {
    float  anomaly_score;    /* reconstruction MSE */
    bool   is_anomalous;     /* score > threshold */
    float  threshold;        /* current threshold in use */
    bool   valid;            /* false if inference failed */
} HCResult;

/**
 * Initialise: mount models partition, load tflite model,
 * allocate tensor arena, load threshold from NVS.
 * Returns 0 on success.
 */
int  hc_init(void);

/**
 * Run inference on one 10-element feature vector.
 * features: float32[HC_N_FEATURES], may contain NaN (imputed internally)
 */
HCResult hc_infer(const float* features);

/**
 * Update the anomaly threshold (called during personal calibration).
 * Persists to NVS.
 */
void hc_set_threshold(float threshold);

/**
 * Get current threshold.
 */
float hc_get_threshold(void);

#ifdef __cplusplus
}
#endif