#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Head F: Circadian Phase Tracker — Forger (1999) ODE.
 * Driven by sleep/wake sequence from Head E.
 * No model file required.
 */

#define HF_BASELINE_DAYS    30
#define HF_MIN_DAYS         7     /* minimum history for Z-score */
#define HF_LUX_WAKE        250.0f
#define HF_LUX_SLEEP         0.0f
#define HF_DT_H         (1.0f/60.0f)   /* 1-minute integration step */

typedef struct {
    float x;    /* circadian phase variable */
    float xc;   /* amplitude variable */
} HFODEState;

typedef struct {
    float dlmo_h;        /* estimated DLMO, hours from midnight */
    float midsleep_h;    /* from Head E */
    float phase_advance; /* vs personal mean, hours (+ve = advanced) */
    float phase_z;       /* Z-score vs personal 30-day baseline */
    bool  valid;         /* false until HF_MIN_DAYS of history */
} HFNightResult;

/**
 * Initialise ODE state and history buffer.
 */
void hf_init(void);

/**
 * Process one day. Call once per night after Head E produces a summary.
 *
 * sleep_wake_minutes: bool[1440], true = sleep for that minute of the day
 * midsleep_h:         from HENightSummary.midsleep_h
 *
 * Returns HFNightResult. valid=false until HF_MIN_DAYS nights processed.
 */
HFNightResult hf_update(const bool* sleep_wake_minutes, float midsleep_h);

/**
 * Reset all state (new user / factory reset).
 */
void hf_reset(void);

#ifdef __cplusplus
}
#endif