#include "feature_extractor.h"
#include "fft_wrapper.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <alloca.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Utilities ───────────────────────────────────────────────────────────── */

static inline float nanval(void) { return __builtin_nanf(""); }

static float mean_f(const float* x, int n)
{
    if (n <= 0) return 0.0f;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return (float)(s / n);
}

static float std_f(const float* x, int n, float m)
{
    if (n <= 1) return 0.0f;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - m;
        s += d * d;
    }
    return (float)sqrtf((float)(s / n));
}

static void moving_average(const float* src, float* dst, int n, int win)
{
    double csum = 0.0;
    for (int i = 0; i < n; i++) {
        csum += src[i];
        int lo = i - win + 1;
        if (lo < 0) lo = 0;
        if (lo > 0) csum -= src[lo - 1];
        dst[i] = (float)(csum / (i - (lo > 0 ? lo - 1 : 0) + 1));
    }
}

/* ── Initialisation ──────────────────────────────────────────────────────── */

void fe_init(FEFilterState* fs)
{
    iir1_reset(&fs->eda_tonic_lp);
    fs->eda_tonic_lp.alpha =
        iir1_alpha_lowpass(FE_EDA_TONIC_CUTOFF, (float)FE_EDA_FS);
}

/* ── ENMO ────────────────────────────────────────────────────────────────── */

void fe_enmo(const float* acc_xyz, float* enmo, int n)
{
    for (int i = 0; i < n; i++) {
        float x = acc_xyz[3 * i + 0];
        float y = acc_xyz[3 * i + 1];
        float z = acc_xyz[3 * i + 2];
        float mag = sqrtf(x * x + y * y + z * z);
        float e = mag - 1.0f;
        enmo[i] = (e > 0.0f) ? e : 0.0f;
    }
}

/* ── BVP peak detection (Elgendi two-MA algorithm) ───────────────────────── */

static int detect_peaks(const float* bvp, int n, int fs,
                        int* peaks, int max_peaks)
{
    /* Step 1: bandpass 0.5-8 Hz */
    float* filt = (float*)alloca(n * sizeof(float));
    memset(filt, 0, n * sizeof(float));
    float* tmp = (float*)alloca(n * sizeof(float));
    memset(tmp,  0, n * sizeof(float));

    iir1_bandpass_buf(bvp, filt, tmp,
                      FE_BVP_BAND_LO, FE_BVP_BAND_HI, (float)fs, n);

    /* Step 2: clip to positive, square */
    float* sq = (float*)alloca(n * sizeof(float));
    memset(sq, 0, n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float c = (filt[i] > 0.0f) ? filt[i] : 0.0f;
        sq[i] = c * c;
    }

    /* Step 3: two moving averages */
    int w_peak = (int)(FE_BVP_W_PEAK_S * fs);
    int w_beat = (int)(FE_BVP_W_BEAT_S * fs);
    if (w_peak < 1) w_peak = 1;
    if (w_beat < 1) w_beat = 1;

    float* ma_peak = (float*)alloca(n * sizeof(float));
    memset(ma_peak, 0, n * sizeof(float));
    float* ma_beat = (float*)alloca(n * sizeof(float));
    memset(ma_beat, 0, n * sizeof(float));

    moving_average(sq, ma_peak, n, w_peak);
    moving_average(sq, ma_beat, n, w_beat);

    /* Step 4: blocks of interest where ma_peak > ma_beat */
    int refractory = (int)(FE_BVP_REFRAC_S * fs);
    int n_peaks    = 0;
    int last_peak  = -refractory * 2;
    int i          = 0;

    while (i < n) {
        if (ma_peak[i] <= ma_beat[i]) { i++; continue; }

        int j = i;
        while (j < n && ma_peak[j] > ma_beat[j]) j++;
        int block_len = j - i;

        if (block_len >= w_peak) {
            int   peak_idx = i;
            float peak_val = filt[i];
            for (int k = i + 1; k < j; k++) {
                if (filt[k] > peak_val) { peak_val = filt[k]; peak_idx = k; }
            }
            if (peak_idx - last_peak >= refractory && n_peaks < max_peaks) {
                peaks[n_peaks++] = peak_idx;
                last_peak = peak_idx;
            }
        }
        i = j;
    }
    return n_peaks;
}

/* ── Cardiac features ────────────────────────────────────────────────────── */

bool fe_cardiac(const float* bvp, int bvp_n, int bvp_fs,
                const float* acc_enmo, int acc_n,
                float* hr_bpm, float* rmssd_ms, float* sdnn_ms)
{
    *hr_bpm   = nanval();
    *rmssd_ms = nanval();
    *sdnn_ms  = nanval();

    /* Motion gate */
    if (acc_enmo != NULL && acc_n > 0) {
        float m = mean_f(acc_enmo, acc_n);
        float s = std_f(acc_enmo, acc_n, m);
        if (s > FE_MOTION_GATE_G) return false;
    }

    int   max_peaks = bvp_n / 10;
    int*  peaks     = (int*)alloca(max_peaks * sizeof(int));
    memset(peaks, 0, max_peaks * sizeof(int));

    int n_peaks = detect_peaks(bvp, bvp_n, bvp_fs, peaks, max_peaks);
    if (n_peaks < 10) return false;

    /* IBIs in ms */
    int    n_ibi = n_peaks - 1;
    float* ibi   = (float*)alloca(n_ibi * sizeof(float));
    memset(ibi, 0, n_ibi * sizeof(float));
    for (int i = 0; i < n_ibi; i++) {
        ibi[i] = (float)(peaks[i + 1] - peaks[i]) * (1000.0f / bvp_fs);
    }

    /* Stage 1: physiological plausibility */
    float* ibi2  = (float*)alloca(n_ibi * sizeof(float));
    memset(ibi2, 0, n_ibi * sizeof(float));
    int n_ibi2 = 0;
    for (int i = 0; i < n_ibi; i++) {
        if (ibi[i] >= FE_IBI_MIN_MS && ibi[i] <= FE_IBI_MAX_MS)
            ibi2[n_ibi2++] = ibi[i];
    }
    if (n_ibi2 < 10) return false;

    /* Stage 2: Berntson local-median filter */
    bool*  keep = (bool*)alloca(n_ibi2 * sizeof(bool));
    memset(keep, 0, n_ibi2 * sizeof(bool));
    float* nbuf = (float*)alloca(5 * sizeof(float));
    memset(nbuf, 0, 5 * sizeof(float));

    for (int i = 0; i < n_ibi2; i++) {
        int lo = i - 2; if (lo < 0) lo = 0;
        int hi = i + 3; if (hi > n_ibi2) hi = n_ibi2;
        int m  = hi - lo;
        memcpy(nbuf, ibi2 + lo, m * sizeof(float));
        for (int a = 1; a < m; a++) {
            float v = nbuf[a]; int b = a - 1;
            while (b >= 0 && nbuf[b] > v) { nbuf[b + 1] = nbuf[b]; b--; }
            nbuf[b + 1] = v;
        }
        float med = nbuf[m / 2];
        float dev = fabsf(ibi2[i] - med) / (med > 0.0f ? med : 1.0f);
        keep[i] = (dev <= FE_IBI_OUTLIER_PCT);
    }

    /* Collect clean IBIs */
    float* ibi_c    = (float*)alloca(n_ibi2 * sizeof(float));
    memset(ibi_c, 0, n_ibi2 * sizeof(float));
    int*   ibi_cidx = (int*)alloca(n_ibi2 * sizeof(int));
    memset(ibi_cidx, 0, n_ibi2 * sizeof(int));
    int n_clean = 0;
    for (int i = 0; i < n_ibi2; i++) {
        if (keep[i]) {
            ibi_c[n_clean]    = ibi2[i];
            ibi_cidx[n_clean] = i;
            n_clean++;
        }
    }
    if (n_clean < FE_IBI_MIN_CLEAN) return false;

    float mean_ibi = mean_f(ibi_c, n_clean);
    *hr_bpm  = 60000.0f / mean_ibi;
    *sdnn_ms = std_f(ibi_c, n_clean, mean_ibi);

    double sum_sq = 0.0;
    int    n_diff = 0;
    for (int i = 1; i < n_clean; i++) {
        if (ibi_cidx[i] == ibi_cidx[i - 1] + 1) {
            double d = ibi_c[i] - ibi_c[i - 1];
            sum_sq += d * d;
            n_diff++;
        }
    }
    *rmssd_ms = (n_diff > 0)
                ? (float)sqrtf((float)(sum_sq / n_diff))
                : nanval();

    return true;
}

/* ── EDA features ────────────────────────────────────────────────────────── */

void fe_eda(const float* eda, int n, int eda_fs,
            FEFilterState* fs,
            float* scl_us, int* scr_count)
{
    *scl_us    = nanval();
    *scr_count = 0;

    if (n < eda_fs * 2) return;

    float* tonic  = (float*)alloca(n * sizeof(float));
    memset(tonic,  0, n * sizeof(float));
    float* phasic = (float*)alloca(n * sizeof(float));
    memset(phasic, 0, n * sizeof(float));

    iir1_lowpass_buf(&fs->eda_tonic_lp, eda, tonic, n);

    float scl_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        phasic[i] = eda[i] - tonic[i];
        scl_sum  += tonic[i];
    }
    *scl_us = scl_sum / (float)n;

    int   n_scr    = 0;
    bool  in_peak  = false;
    float peak_amp = 0.0f;

    for (int i = 1; i < n; i++) {
        float dp_prev = phasic[i - 1] - (i >= 2 ? phasic[i - 2] : phasic[i - 1]);
        float dp_curr = phasic[i]     - phasic[i - 1];

        if (dp_prev > 0.0f && dp_curr <= 0.0f) {
            if (!in_peak) { peak_amp = phasic[i]; in_peak = true; }
            else if (phasic[i] > peak_amp) peak_amp = phasic[i];
        } else if (dp_prev < 0.0f && dp_curr >= 0.0f && in_peak) {
            if (peak_amp > FE_EDA_SCR_THRESH) n_scr++;
            in_peak  = false;
            peak_amp = 0.0f;
        }
    }
    *scr_count = n_scr;
}

/* ── Top-level window extraction ─────────────────────────────────────────── */

void fe_extract_window(
    const float* acc,
    const float* bvp,
    const float* eda,
    const float* temp,
    FEFilterState* fs,
    float* out)
{
    for (int i = 0; i < FE_N_FEATURES; i++) out[i] = nanval();

    /* ACC */
    static float enmo[FE_ACC_WINDOW_N];
    fe_enmo(acc, enmo, FE_ACC_WINDOW_N);

    float enmo_mean = mean_f(enmo, FE_ACC_WINDOW_N);
    float enmo_std  = std_f(enmo, FE_ACC_WINDOW_N, enmo_mean);
    out[0] = enmo_mean;
    out[1] = enmo_std;

    if (enmo_mean >= FE_STILL_THRESH_G) {
        static float power[FE_FFT_N / 2 + 1];
        static float work [FE_FFT_N * 2];
        const float* fft_src = enmo + FE_ACC_WINDOW_N - FE_FFT_N;
        fft_power_spectrum(fft_src, power, FE_FFT_N, work);
        FFTBandResult r = fft_band_dominant(power, FE_FFT_N,
                                            (float)FE_ACC_FS, 0.5f, 16.0f);
        out[2] = r.freq_hz;
        out[3] = r.rel_power;
    }

    /* Cardiac */
    float hr, rmssd, sdnn;
    fe_cardiac(bvp, FE_BVP_WINDOW_N, FE_BVP_FS,
               enmo, FE_ACC_WINDOW_N,
               &hr, &rmssd, &sdnn);
    out[4] = hr;
    out[5] = rmssd;
    out[6] = sdnn;

    /* EDA */
    float scl;
    int   scr;
    fe_eda(eda, FE_EDA_WINDOW_N, FE_EDA_FS, fs, &scl, &scr);
    out[7] = scl;
    out[8] = (float)scr;

    /* Temperature */
    if (temp != NULL)
        out[9] = mean_f(temp, FE_EDA_WINDOW_N);
}