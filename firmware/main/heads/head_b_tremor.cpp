#include "head_b_tremor.h"
#include "fft_wrapper.h"
#include <math.h>
#include <string.h>

static int s_consecutive = 0;

void hb_init(void)  { hb_reset(); }
void hb_reset(void) { s_consecutive = 0; }

HBResult hb_classify(const float* acc_xyz)
{
    HBResult res = {0, 0.0f, 0.0f, 0.0f, 0.0f, false};

    /* Mean-remove each axis (removes gravity / DC) */
    float seg[HB_N_FFT * 3];
    memcpy(seg, acc_xyz, sizeof(seg));

    float mean[3] = {0};
    for (int i = 0; i < HB_N_FFT; i++) {
        mean[0] += seg[3*i+0];
        mean[1] += seg[3*i+1];
        mean[2] += seg[3*i+2];
    }
    for (int a = 0; a < 3; a++) mean[a] /= HB_N_FFT;
    for (int i = 0; i < HB_N_FFT; i++) {
        seg[3*i+0] -= mean[0];
        seg[3*i+1] -= mean[1];
        seg[3*i+2] -= mean[2];
    }

    /* RMS across all axes */
    float rms_sum = 0.0f;
    for (int i = 0; i < HB_N_FFT * 3; i++) rms_sum += seg[i] * seg[i];
    float rms = sqrtf(rms_sum / (HB_N_FFT * 3));
    res.rms = rms;

    /* Motion gate */
    if (rms < HB_MOTION_MIN_RMS || rms > HB_MOTION_MAX_RMS) {
        s_consecutive = 0;
        return res;   /* label 0 */
    }

    /* FFT each axis, sum power spectra */
    static float power[HB_N_FFT / 2 + 1];
    static float work [HB_N_FFT * 2];
    static float axis_buf[HB_N_FFT];

    memset(power, 0, sizeof(power));
    static float axis_power[HB_N_FFT / 2 + 1];

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < HB_N_FFT; i++) axis_buf[i] = seg[3*i+a];
        fft_power_spectrum(axis_buf, axis_power, HB_N_FFT, work);
        for (int k = 0; k <= HB_N_FFT/2; k++) power[k] += axis_power[k];
    }

    /* Relative band powers */
    float pd_p = fft_relative_band_power(power, HB_N_FFT,
                                          HB_ACC_FS, HB_PD_LO, HB_PD_HI);
    float et_p = fft_relative_band_power(power, HB_N_FFT,
                                          HB_ACC_FS, HB_ET_LO, HB_ET_HI);

    /* Dominant frequency */
    FFTBandResult pd_r = fft_band_dominant(power, HB_N_FFT, HB_ACC_FS,
                                            HB_PD_LO, HB_PD_HI);
    FFTBandResult et_r = fft_band_dominant(power, HB_N_FFT, HB_ACC_FS,
                                            HB_ET_LO, HB_ET_HI);
    res.pd_power   = pd_p;
    res.et_power   = et_p;
    res.dom_freq_hz = (pd_r.band_power >= et_r.band_power)
                      ? pd_r.freq_hz : et_r.freq_hz;

    /* Classification */
    if      (pd_p > HB_PD_THRESH && rms < HB_REST_THRESH_RMS) res.label = 1;
    else if (et_p > HB_ET_THRESH)                              res.label = 2;
    else if (pd_p > HB_INDET_THRESH || et_p > HB_INDET_THRESH) res.label = 3;
    else                                                        res.label = 0;

    /* Sustained flag */
    if (res.label == 1 || res.label == 2) {
        s_consecutive++;
        res.sustained = (s_consecutive >= HB_MIN_CONSECUTIVE);
    } else {
        s_consecutive = 0;
    }

    return res;
}