#include "head_f_circadian.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Forger 1999 constants ───────────────────────────────────────────────── */
static const float kMu     = 0.13f;
static const float kQ      = 1.0f / 3.0f;
static const float kK      = 0.55f;
static const float kAlpha0 = 0.05f;
static const float kP      = 0.50f;
static const float kI0     = 9500.0f;
static const float kDlmoOffset = 7.0f;   /* CBTmin = DLMO + 7 h */

/* ── State ───────────────────────────────────────────────────────────────── */
static HFODEState s_ode;
static float      s_history[HF_BASELINE_DAYS];
static int        s_hist_n    = 0;
static int        s_hist_head = 0;   /* ring buffer index */

void hf_init(void)  { hf_reset(); }
void hf_reset(void)
{
    s_ode.x = 0.0f; s_ode.xc = 0.1f;
    memset(s_history, 0, sizeof(s_history));
    s_hist_n = 0; s_hist_head = 0;
}

/* ── Photic drive ────────────────────────────────────────────────────────── */
static float alpha(float lux)
{
    float lp = powf(lux, kP);
    float i0p = powf(kI0, kP);
    return kAlpha0 * lp / (lp + i0p);
}

/* ── Euler step ──────────────────────────────────────────────────────────── */
static void ode_step(HFODEState* s, float lux, float dt)
{
    float a  = alpha(lux);
    float B  = a * (-kQ * s->xc + a * (1.0f - (kP / kK) * s->x));
    float r2 = s->x * s->x + s->xc * s->xc;
    float dx  = ((float)M_PI / 12.0f) * (s->xc - kMu * s->x * r2 + B);
    float dxc = ((float)M_PI / 12.0f) * (-s->x - kMu * s->xc * r2);
    s->x  += dx  * dt;
    s->xc += dxc * dt;
}

/* ── One-day integration ─────────────────────────────────────────────────── */
static float integrate_day(const bool* sleep_wake_minutes)
{
    float xc_min = 1e9f;
    int   xc_min_minute = 0;

    for (int m = 0; m < 1440; m++) {
        float lux = sleep_wake_minutes[m] ? HF_LUX_SLEEP : HF_LUX_WAKE;
        ode_step(&s_ode, lux, HF_DT_H);
        if (s_ode.xc < xc_min) { xc_min = s_ode.xc; xc_min_minute = m; }
    }

    /* CBTmin -> DLMO */
    float cbtmin_h = xc_min_minute / 60.0f;
    float dlmo_h   = fmodf(cbtmin_h - kDlmoOffset + 24.0f, 24.0f);
    return dlmo_h;
}

/* ── Public update ───────────────────────────────────────────────────────── */
HFNightResult hf_update(const bool* sleep_wake_minutes, float midsleep_h)
{
    HFNightResult res = {0};
    float dlmo_h = integrate_day(sleep_wake_minutes);

    /* Push into ring buffer */
    s_history[s_hist_head] = dlmo_h;
    s_hist_head = (s_hist_head + 1) % HF_BASELINE_DAYS;
    if (s_hist_n < HF_BASELINE_DAYS) s_hist_n++;

    res.dlmo_h    = dlmo_h;
    res.midsleep_h = midsleep_h;

    if (s_hist_n < HF_MIN_DAYS) { res.valid = false; return res; }

    /* Personal baseline statistics */
    float mean = 0.0f;
    for (int i = 0; i < s_hist_n; i++) mean += s_history[i];
    mean /= s_hist_n;

    float var = 0.0f;
    for (int i = 0; i < s_hist_n; i++) {
        float d = s_history[i] - mean; var += d * d;
    }
    float sd = sqrtf(var / s_hist_n);

    /* Wrap advance to [-12, 12] */
    float adv = dlmo_h - mean;
    adv = fmodf(adv + 12.0f, 24.0f) - 12.0f;

    res.phase_advance = adv;
    res.phase_z       = (sd > 0.01f) ? (adv / sd) : 0.0f;
    res.valid         = true;
    return res;
}