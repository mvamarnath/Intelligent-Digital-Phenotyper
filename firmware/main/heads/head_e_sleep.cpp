#include "head_e_sleep.h"
#include "feature_extractor.h"   /* for fe_enmo() */
#include <math.h>
#include <string.h>

/* ── Causal median-filter ring buffer ────────────────────────────────────── */
static bool  s_ring[HE_SMOOTH_EPOCHS];
static int   s_ring_head = 0;
static int   s_ring_n    = 0;   /* how many valid entries */

void he_init(void)  { he_reset(); }
void he_reset(void) { memset(s_ring, 0, sizeof(s_ring)); s_ring_head = 0; s_ring_n = 0; }

static bool median_bool(const bool* arr, int n)
{
    int ones = 0;
    for (int i = 0; i < n; i++) if (arr[i]) ones++;
    return (ones > n / 2);
}

/* ── Epoch classification ────────────────────────────────────────────────── */
HEEpochResult he_classify_epoch(const float* acc_xyz)
{
    HEEpochResult res = {false, 0.0f, 0.0f};

    /* ENMO per sample */
    float enmo[HE_SAMPLES_PER_EPOCH];
    fe_enmo(acc_xyz, enmo, HE_SAMPLES_PER_EPOCH);

    /* Mean ENMO over the full epoch */
    float sum = 0.0f;
    for (int i = 0; i < HE_SAMPLES_PER_EPOCH; i++) sum += enmo[i];
    res.enmo_mean = sum / HE_SAMPLES_PER_EPOCH;

    /* 5-second block means */
    int   samps_per_block = HE_BLOCK_S * HE_ACC_FS;   /* 160 */
    int   n_blocks = HE_SAMPLES_PER_EPOCH / samps_per_block;  /* 6 */
    float block_mean[6];
    for (int b = 0; b < n_blocks; b++) {
        float bs = 0.0f;
        for (int s = 0; s < samps_per_block; s++)
            bs += enmo[b * samps_per_block + s];
        block_mean[b] = bs / samps_per_block;
    }

    /* Mean absolute difference between consecutive blocks */
    float diff_sum = 0.0f;
    for (int b = 1; b < n_blocks; b++)
        diff_sum += fabsf(block_mean[b] - block_mean[b - 1]);
    res.enmo_diff = diff_sum / (n_blocks - 1);

    /* Raw classification */
    bool raw_sleep = (res.enmo_mean < HE_THRESH_ENMO) &&
                     (res.enmo_diff < HE_THRESH_DIFF);

    /* Insert into ring buffer */
    s_ring[s_ring_head] = raw_sleep;
    s_ring_head = (s_ring_head + 1) % HE_SMOOTH_EPOCHS;
    if (s_ring_n < HE_SMOOTH_EPOCHS) s_ring_n++;

    /* Causal median filter over available entries */
    res.is_sleep = median_bool(s_ring, s_ring_n);
    return res;
}

/* ── Night summary ───────────────────────────────────────────────────────── */
HENightSummary he_summarise_night(const HEEpochResult* epochs,
                                  int n_epochs,
                                  float start_hour)
{
    HENightSummary s = {};

    /* Find first and last sleep epoch */
    int first_sleep = -1, last_sleep = -1;
    for (int i = 0; i < n_epochs; i++) {
        if (epochs[i].is_sleep) {
            if (first_sleep < 0) first_sleep = i;
            last_sleep = i;
        }
    }

    float epoch_h = HE_EPOCH_S / 3600.0f;
    int min_sleep_epochs = (int)(60.0f * 60.0f / HE_EPOCH_S); /* 60 min */

    int tib_epochs  = (last_sleep - first_sleep + 1);
    if (first_sleep < 0 || tib_epochs < min_sleep_epochs) {
        s.valid = false;
        return s;
    }

    int sleep_count = 0;
    int n_awakenings = 0;
    bool prev_sleep = true;
    for (int i = first_sleep; i <= last_sleep; i++) {
        if (epochs[i].is_sleep) {
            sleep_count++;
            prev_sleep = true;
        } else {
            if (prev_sleep) n_awakenings++;
            prev_sleep = false;
        }
    }

    s.onset_h      = start_hour + first_sleep * epoch_h;
    s.wake_h       = start_hour + (last_sleep + 1) * epoch_h;
    s.duration_h   = sleep_count * epoch_h;
    s.efficiency   = (float)sleep_count / (float)tib_epochs;
    s.midsleep_h   = (s.onset_h + s.wake_h) / 2.0f;
    s.n_awakenings = n_awakenings;
    s.valid        = true;
    return s;
}