#include "iir_filter.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Alpha computation ───────────────────────────────────────────────────── */

float iir1_alpha_lowpass(float fc_hz, float fs_hz)
{
    float rc = 1.0f / (2.0f * (float)M_PI * fc_hz);
    float dt = 1.0f / fs_hz;
    return dt / (rc + dt);
}

float iir1_alpha_highpass(float fc_hz, float fs_hz)
{
    float rc = 1.0f / (2.0f * (float)M_PI * fc_hz);
    float dt = 1.0f / fs_hz;
    return rc / (rc + dt);
}

/* ── State management ────────────────────────────────────────────────────── */

void iir1_reset(IIR1State* s)
{
    s->y_prev = 0.0f;
}

/* ── Per-sample processing ───────────────────────────────────────────────── */

float iir1_lowpass_sample(IIR1State* s, float x)
{
    float y = s->alpha * x + (1.0f - s->alpha) * s->y_prev;
    s->y_prev = y;
    return y;
}

float iir1_highpass_sample(IIR1State* s, float x, float x_prev)
{
    float y = s->alpha * (s->y_prev + x - x_prev);
    s->y_prev = y;
    return y;
}

/* ── Buffer processing ───────────────────────────────────────────────────── */

void iir1_lowpass_buf(IIR1State* s, const float* src, float* dst, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i] = s->alpha * src[i] + (1.0f - s->alpha) * s->y_prev;
        s->y_prev = dst[i];
    }
}

void iir1_highpass_buf(IIR1State* s, const float* src, float* dst, int n)
{
    if (n == 0) return;
    /* First sample: no previous input, so y[0] = 0 */
    dst[0] = 0.0f;
    s->y_prev = 0.0f;
    for (int i = 1; i < n; i++) {
        float y = s->alpha * (s->y_prev + src[i] - src[i - 1]);
        s->y_prev = y;
        dst[i] = y;
    }
}

void iir1_bandpass_buf(const float* src, float* dst, float* tmp,
                       float lo_hz, float hi_hz, float fs_hz, int n)
{
    /* Stage 1: high-pass at lo_hz -> tmp */
    IIR1State hp = { .alpha = iir1_alpha_highpass(lo_hz, fs_hz), .y_prev = 0.0f };
    iir1_highpass_buf(&hp, src, tmp, n);

    /* Stage 2: low-pass at hi_hz -> dst */
    IIR1State lp = { .alpha = iir1_alpha_lowpass(hi_hz, fs_hz), .y_prev = 0.0f };
    iir1_lowpass_buf(&lp, tmp, dst, n);
}