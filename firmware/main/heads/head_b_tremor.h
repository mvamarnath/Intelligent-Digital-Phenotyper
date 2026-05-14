#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Head B: Tremor Classifier — spectral rule engine.
 *
 * Classes:
 *   0 = no_tremor
 *   1 = PD_tremor    (4-6 Hz resting)
 *   2 = ET_tremor    (8-12 Hz action)
 *   3 = indeterminate
 */

#define HB_N_FFT            128      /* 4 s at 32 Hz */
#define HB_ACC_FS           32
#define HB_SAMPLES          HB_N_FFT

#define HB_PD_LO            4.0f
#define HB_PD_HI            6.0f
#define HB_ET_LO            8.0f
#define HB_ET_HI           12.0f

#define HB_PD_THRESH        0.35f
#define HB_ET_THRESH        0.35f
#define HB_INDET_THRESH     0.20f
#define HB_REST_THRESH_RMS  0.05f
#define HB_MOTION_MIN_RMS   0.003f
#define HB_MOTION_MAX_RMS   0.15f
#define HB_MIN_CONSECUTIVE  6        /* 24 s sustained = clinical flag */

typedef struct {
    int   label;          /* 0-3 */
    float pd_power;       /* relative spectral power in PD band */
    float et_power;       /* relative spectral power in ET band */
    float dom_freq_hz;    /* dominant frequency in tremor bands */
    float rms;            /* AC RMS of the window */
    bool  sustained;      /* true after HB_MIN_CONSECUTIVE consecutive flags */
} HBResult;

void     hb_init(void);
HBResult hb_classify(const float* acc_xyz);  /* acc_xyz: float[HB_N_FFT*3] */
void     hb_reset(void);

#ifdef __cplusplus
}
#endif