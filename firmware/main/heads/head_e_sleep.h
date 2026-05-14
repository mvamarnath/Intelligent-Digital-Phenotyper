#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Head E: Sleep/Wake Classifier — van Hees HDCZA algorithm.
 * Operates on 30-second ACC epochs.
 * No model file required — purely deterministic.
 */

#define HE_EPOCH_S          30
#define HE_ACC_FS           32
#define HE_SAMPLES_PER_EPOCH (HE_EPOCH_S * HE_ACC_FS)   /* 960 */
#define HE_SMOOTH_EPOCHS     5

/* Published van Hees thresholds */
#define HE_THRESH_ENMO      0.013f   /* g */
#define HE_THRESH_DIFF      0.013f   /* g */
#define HE_BLOCK_S          5        /* seconds per 5-s ENMO block */

typedef struct {
    bool  is_sleep;          /* true = sleep epoch */
    float enmo_mean;         /* mean ENMO for this epoch */
    float enmo_diff;         /* mean abs-diff of 5-s ENMO blocks */
} HEEpochResult;

typedef struct {
    float  onset_h;          /* sleep onset, hours from midnight */
    float  wake_h;           /* wake time, hours from midnight */
    float  duration_h;       /* sleep duration */
    float  efficiency;       /* fraction of TIB classified sleep */
    float  midsleep_h;       /* midpoint (fed to Head F) */
    int    n_awakenings;
    bool   valid;
} HENightSummary;

/**
 * Initialise the sleep classifier and its median-filter ring buffer.
 */
void he_init(void);

/**
 * Process one 30-second ACC epoch.
 * acc_xyz: float32[HE_SAMPLES_PER_EPOCH * 3], row-major (x,y,z per sample)
 *
 * Internally maintains a 5-epoch causal median filter.
 * Returns the smoothed sleep/wake classification.
 */
HEEpochResult he_classify_epoch(const float* acc_xyz);

/**
 * Process a night of epochs and extract sleep summary.
 * epochs:     array of HEEpochResult (from he_classify_epoch)
 * n_epochs:   number of epochs
 * start_hour: hour-of-day of the first epoch (e.g. 21.5 = 21:30)
 */
HENightSummary he_summarise_night(const HEEpochResult* epochs,
                                  int n_epochs,
                                  float start_hour);

/**
 * Reset internal ring buffer (call at start of each recording session).
 */
void he_reset(void);

#ifdef __cplusplus
}
#endif