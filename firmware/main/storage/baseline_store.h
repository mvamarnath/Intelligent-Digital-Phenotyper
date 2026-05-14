#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Baseline Store — persists per-user rolling statistics to SPIFFS storage
 * partition. Stores mean and std for each of the 10 features over a
 * rolling 30-day window. Used to compute personal Z-scores on device.
 *
 * Also stores the autoencoder anomaly threshold (95th percentile of
 * baseline reconstruction errors) once the calibration period completes.
 */

#define BS_N_FEATURES       10
#define BS_WINDOW_DAYS      30
#define BS_CALIB_DAYS        7    /* days before baseline is considered valid */
#define BS_MOUNT_POINT      "/storage"
#define BS_FILE_PATH        "/storage/baseline.bin"
#define BS_NVS_NS           "baseline"

typedef struct __attribute__((packed)) {
    uint32_t magic;              /* 0x42534C4E "BSLN" */
    uint32_t n_windows;          /* total windows processed */
    float    mean[BS_N_FEATURES];
    float    m2[BS_N_FEATURES];  /* Welford M2 for online variance */
    float    ae_threshold;       /* autoencoder anomaly threshold */
    uint32_t calibration_days;   /* days of data collected */
    bool     is_valid;           /* true after BS_CALIB_DAYS */
    uint8_t  reserved[159];      /* pad to 256 bytes: 256 - 97 actual bytes */
} BSData;

static_assert(sizeof(BSData) == 256, "BSData must be 256 bytes");

/**
 * Mount storage partition and load baseline from file.
 * Creates fresh baseline if file absent.
 */
int  bs_init(void);

/**
 * Update the rolling baseline with one new feature vector.
 * Uses Welford online algorithm — O(1) memory, O(1) time.
 */
void bs_update(const float* features);

/**
 * Compute Z-score for each feature vs current baseline.
 * out_z: float[BS_N_FEATURES] — NaN if baseline not yet valid.
 */
void bs_zscore(const float* features, float* out_z);

/**
 * Get current baseline data (read-only copy).
 */
BSData bs_get(void);

/**
 * Persist current baseline to flash.
 * Call once per hour or after each daily summary.
 */
int bs_save(void);

/**
 * Set the autoencoder anomaly threshold after calibration.
 */
void bs_set_ae_threshold(float threshold);

/**
 * True if enough data has been collected for reliable Z-scores.
 */
bool bs_is_valid(void);

/**
 * Erase baseline (factory reset).
 */
int bs_erase(void);

#ifdef __cplusplus
}
#endif