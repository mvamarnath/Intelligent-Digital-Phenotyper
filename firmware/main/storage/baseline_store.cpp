#include "baseline_store.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

static const char* TAG = "BS";
static BSData s_data;
static bool   s_ready = false;

#define BS_MAGIC 0x42534C4EUL

static int mount_spiffs(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path              = BS_MOUNT_POINT,
        .partition_label        = "storage",
        .max_files              = 4,
        .format_if_mount_failed = true,
    };
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    return (ret == ESP_OK || ret == ESP_ERR_INVALID_STATE) ? 0 : -1;
}

static void init_fresh(void)
{
    memset(&s_data, 0, sizeof(s_data));
    s_data.magic             = BS_MAGIC;
    s_data.ae_threshold      = 2.0f;   /* conservative default */
    s_data.is_valid          = false;
    s_data.calibration_days  = 0;
    s_data.n_windows         = 0;
}

int bs_init(void)
{
    if (mount_spiffs() != 0) return -1;

    FILE* f = fopen(BS_FILE_PATH, "rb");
    if (f) {
        size_t n = fread(&s_data, 1, sizeof(s_data), f);
        fclose(f);
        if (n == sizeof(s_data) && s_data.magic == BS_MAGIC) {
            ESP_LOGI(TAG, "Loaded baseline. Windows=%lu Valid=%d",
                     (unsigned long)s_data.n_windows, s_data.is_valid);
            s_ready = true;
            return 0;
        }
        ESP_LOGW(TAG, "Corrupt baseline — reinitialising");
    }

    init_fresh();
    s_ready = true;
    bs_save();
    ESP_LOGI(TAG, "Fresh baseline created");
    return 0;
}

/* ── Welford online mean/variance ────────────────────────────────────────── */
void bs_update(const float* features)
{
    if (!s_ready) return;
    s_data.n_windows++;

    for (int i = 0; i < BS_N_FEATURES; i++) {
        float x = features[i];
        if (x != x) continue;   /* skip NaN */
        float delta  = x - s_data.mean[i];
        s_data.mean[i] += delta / (float)s_data.n_windows;
        float delta2 = x - s_data.mean[i];
        s_data.m2[i] += delta * delta2;
    }

    /* One window = 60 s; 1440 windows = 1 day */
    if (s_data.n_windows % 1440 == 0) {
        s_data.calibration_days++;
        if (s_data.calibration_days >= BS_CALIB_DAYS && !s_data.is_valid) {
            s_data.is_valid = true;
            ESP_LOGI(TAG, "Baseline now valid after %lu days",
                     (unsigned long)s_data.calibration_days);
        }
    }
}

void bs_zscore(const float* features, float* out_z)
{
    for (int i = 0; i < BS_N_FEATURES; i++) {
        if (!s_data.is_valid || s_data.n_windows < 2) {
            out_z[i] = __builtin_nanf("");
            continue;
        }
        float var = s_data.m2[i] / (float)(s_data.n_windows - 1);
        float sd  = sqrtf(var);
        if (sd < 1e-6f) { out_z[i] = 0.0f; continue; }
        float x = features[i];
        if (x != x) { out_z[i] = __builtin_nanf(""); continue; }
        out_z[i] = (x - s_data.mean[i]) / sd;
    }
}

BSData bs_get(void)  { return s_data; }

bool bs_is_valid(void) { return s_ready && s_data.is_valid; }

void bs_set_ae_threshold(float t)
{
    s_data.ae_threshold = t;
}

int bs_save(void)
{
    if (!s_ready) return -1;
    FILE* f = fopen(BS_FILE_PATH, "wb");
    if (!f) { ESP_LOGE(TAG, "Cannot save baseline"); return -1; }
    fwrite(&s_data, 1, sizeof(s_data), f);
    fclose(f);
    ESP_LOGD(TAG, "Baseline saved");
    return 0;
}

int bs_erase(void)
{
    init_fresh();
    return bs_save();
}