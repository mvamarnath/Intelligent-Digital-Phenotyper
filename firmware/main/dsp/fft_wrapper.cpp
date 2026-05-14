#include "fft_wrapper.h"
#include "esp_dsp.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static const char* TAG = "FFT";

/* Pre-computed Hann windows for the sizes we use */
static float s_hann_128[128];
static float s_hann_256[256];
static float s_hann_512[512];
static bool  s_initialised = false;

static void precompute_hann(float* w, int n)
{
    for (int i = 0; i < n; i++) {
        w[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(n - 1));
    }
}

int fft_init(void)
{
    if (s_initialised) return 0;

    esp_err_t ret = dsps_fft2r_init_fc32(NULL, FFT_MAX_N);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "dsps_fft2r_init_fc32 failed: %d", ret);
        return -1;
    }

    precompute_hann(s_hann_128, 128);
    precompute_hann(s_hann_256, 256);
    precompute_hann(s_hann_512, 512);

    s_initialised = true;
    ESP_LOGI(TAG, "FFT initialised (max N=%d)", FFT_MAX_N);
    return 0;
}

static const float* get_hann(int n)
{
    switch (n) {
        case 128: return s_hann_128;
        case 256: return s_hann_256;
        case 512: return s_hann_512;
        default:  return NULL;   /* caller must handle */
    }
}

void fft_power_spectrum(const float* src, float* power, int n, float* work)
{
    /* work layout: [re0, im0, re1, im1, ...] — interleaved complex */
    const float* hann = get_hann(n);

    /* Remove DC and apply window */
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += src[i];
    mean /= (float)n;

    for (int i = 0; i < n; i++) {
        float w = (hann != NULL) ? hann[i] : 1.0f;
        work[2 * i]     = (src[i] - mean) * w;   /* real */
        work[2 * i + 1] = 0.0f;                   /* imag */
    }

    /* In-place radix-2 FFT (ESP-DSP expects interleaved complex) */
    dsps_fft2r_fc32(work, n);
    dsps_bit_rev_fc32(work, n);

    /* Power spectrum: |X[k]|^2, only positive frequencies */
    int bins = n / 2 + 1;
    for (int k = 0; k < bins; k++) {
        float re = work[2 * k];
        float im = work[2 * k + 1];
        power[k] = re * re + im * im;
    }
}

FFTBandResult fft_band_dominant(const float* power, int n, float fs_hz,
                                float lo_hz, float hi_hz)
{
    FFTBandResult res = {0.0f, 0.0f, 0.0f};
    float freq_res = fs_hz / (float)n;
    int bins = n / 2 + 1;

    float total_power = 0.0f;
    float band_power  = 0.0f;
    float peak_power  = -1.0f;
    int   peak_bin    = -1;

    for (int k = 0; k < bins; k++) {
        float f = k * freq_res;
        total_power += power[k];
        if (f >= lo_hz && f <= hi_hz) {
            band_power += power[k];
            if (power[k] > peak_power) {
                peak_power = power[k];
                peak_bin   = k;
            }
        }
    }

    if (peak_bin < 0 || band_power <= 0.0f) return res;

    res.freq_hz    = peak_bin * freq_res;
    res.band_power = band_power;
    res.rel_power  = (band_power > 0.0f && total_power > 0.0f)
                     ? (power[peak_bin] / band_power)
                     : 0.0f;
    return res;
}

float fft_relative_band_power(const float* power, int n, float fs_hz,
                              float lo_hz, float hi_hz)
{
    float freq_res    = fs_hz / (float)n;
    int   bins        = n / 2 + 1;
    float total_power = 0.0f;
    float band_power  = 0.0f;

    for (int k = 0; k < bins; k++) {
        float f = k * freq_res;
        total_power += power[k];
        if (f >= lo_hz && f <= hi_hz) band_power += power[k];
    }

    return (total_power > 0.0f) ? (band_power / total_power) : 0.0f;
}