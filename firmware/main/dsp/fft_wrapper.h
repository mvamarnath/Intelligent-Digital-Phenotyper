#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FFT wrapper around ESP-DSP dsps_fft2r_fc32.
 *
 * Provides:
 *   - Real-input radix-2 FFT (Hann-windowed)
 *   - Power spectrum computation
 *   - Band power (relative) extraction
 *   - Dominant frequency search within a band
 *
 * All FFT sizes must be powers of two.
 * Maximum supported size: FFT_MAX_N (512 points).
 */

#define FFT_MAX_N       512
#define FFT_ACC_FS_HZ   32      /* accelerometer sample rate */
#define FFT_BVP_FS_HZ   64      /* BVP sample rate */
#define FFT_TREMOR_N    128     /* 4 s at 32 Hz — tremor classifier */
#define FFT_BVP_N       256     /* 4 s at 64 Hz — BVP peak assist */

/**
 * Result of dominant-frequency search.
 */
typedef struct {
    float freq_hz;      /* dominant frequency in the search band */
    float rel_power;    /* power at peak / total band power (0-1) */
    float band_power;   /* total power in band (absolute) */
} FFTBandResult;

/**
 * Initialise ESP-DSP FFT tables. Must be called once at startup.
 * Returns 0 on success, non-zero on error.
 */
int fft_init(void);

/**
 * Compute Hann-windowed real FFT power spectrum.
 *
 * src:      input float32 array, length n (real samples)
 * power:    output float32 array, length n/2+1 (power per bin)
 * n:        FFT size (power of two, <= FFT_MAX_N)
 * work:     scratch buffer, length 2*n floats (caller-provided)
 *
 * Power is not normalised — use rel_power from FFTBandResult for
 * comparisons across windows of different lengths.
 */
void fft_power_spectrum(const float* src, float* power, int n, float* work);

/**
 * Find dominant frequency within [lo_hz, hi_hz].
 *
 * power:    power spectrum from fft_power_spectrum(), length n/2+1
 * n:        FFT size used to compute power
 * fs_hz:    sample rate of the original signal
 * lo_hz:    lower band edge (inclusive)
 * hi_hz:    upper band edge (inclusive)
 *
 * Returns FFTBandResult. If no bins fall in the band, all fields are 0.
 */
FFTBandResult fft_band_dominant(const float* power, int n, float fs_hz,
                                float lo_hz, float hi_hz);

/**
 * Compute relative power in [lo_hz, hi_hz] vs total spectrum power.
 * Returns value in [0, 1]. Returns 0 if total power is zero.
 */
float fft_relative_band_power(const float* power, int n, float fs_hz,
                              float lo_hz, float hi_hz);

#ifdef __cplusplus
}
#endif