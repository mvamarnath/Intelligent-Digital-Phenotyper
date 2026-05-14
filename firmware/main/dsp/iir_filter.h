#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * First-order IIR filters — causal, streaming-compatible.
 * Identical algorithm to features.py lowpass_1pole / highpass_1pole.
 *
 * All operate in-place on float32 arrays.
 * State is maintained externally so the same filter struct can be
 * used across consecutive 60-second windows without resetting.
 */

typedef struct {
    float alpha;   /* filter coefficient */
    float y_prev;  /* previous output sample */
} IIR1State;

/**
 * Compute alpha for a first-order RC low-pass.
 * fc_hz:  -3 dB cutoff frequency
 * fs_hz:  sample rate
 */
float iir1_alpha_lowpass(float fc_hz, float fs_hz);

/**
 * Compute alpha for a first-order RC high-pass.
 */
float iir1_alpha_highpass(float fc_hz, float fs_hz);

/**
 * Initialise state to zero.
 */
void iir1_reset(IIR1State* s);

/**
 * Process one sample through a low-pass filter.
 * y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
 */
float iir1_lowpass_sample(IIR1State* s, float x);

/**
 * Process one sample through a high-pass filter.
 * y[n] = alpha * (y[n-1] + x[n] - x[n-1])
 */
float iir1_highpass_sample(IIR1State* s, float x, float x_prev);

/**
 * Process a full buffer through a low-pass filter.
 * Output written to dst (may equal src for in-place).
 */
void iir1_lowpass_buf(IIR1State* s, const float* src, float* dst, int n);

/**
 * Process a full buffer through a high-pass filter.
 */
void iir1_highpass_buf(IIR1State* s, const float* src, float* dst, int n);

/**
 * Cascade: high-pass at lo_hz then low-pass at hi_hz.
 * Uses two internal states (hp then lp).
 * tmp must be a caller-provided scratch buffer of length n floats.
 */
void iir1_bandpass_buf(const float* src, float* dst, float* tmp,
                       float lo_hz, float hi_hz, float fs_hz, int n);

#ifdef __cplusplus
}
#endif