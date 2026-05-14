/**
 * IDP Firmware — main entry point.
 *
 * Task architecture (dual-core ESP32-S3):
 *
 *   Core 0:
 *     sensor_task    — reads ACC/BVP/EDA/TEMP into circular buffers (32/64/4/4 Hz)
 *     analysis_task  — extracts features every 60 s, runs all heads, logs events
 *
 *   Core 1 (when BLE active):
 *     nimble_host    — managed by NimBLE port
 *
 * Circular buffers are protected by FreeRTOS ring buffer primitives.
 * The sensor_task writes; the analysis_task reads snapshots every 60 s.
 *
 * NOTE: Sensor drivers for MAX30102 (PPG, I2C 0x57),
 *       MPU9250 (ACC, I2C 0x68), and CJMCU-6701 (GSR/EDA, ADC GPIO1)
 *       are fully implemented in main/sensors/.
 *       Replace get_sensor_data() pin assignments in sensor_pins.h
 *       if your hardware layout differs.
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h"
extern "C" {
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_heap_caps.h"
}

#include "feature_extractor.h"
#include "head_b_tremor.h"
#include "head_c_autoencoder.h"
#include "head_d_stress.h"
#include "head_e_sleep.h"
#include "head_f_circadian.h"
#include "event_logger.h"
#include "baseline_store.h"
#include "ble_export.h"
#include <math.h>
#include <string.h>
#include "fft_wrapper.h"
#include "sensors/sensor_pins.h"
#include "sensors/i2c_bus.h"
#include "sensors/max30102.h"
#include "sensors/mpu9250.h"
#include "sensors/gsr_eda.h"

static const char* TAG = "MAIN";

/* ── Buffer sizes ────────────────────────────────────────────────────────── */
#define ACC_BUF_SECONDS     65    /* 65 s at 32 Hz = 2080 samples */
#define BVP_BUF_SECONDS     65    /* 65 s at 64 Hz = 4160 samples */
#define EDA_BUF_SECONDS     65    /* 65 s at  4 Hz =  260 samples */

#define ACC_BUF_N    (ACC_BUF_SECONDS * FE_ACC_FS * 3)   /* x,y,z */
#define BVP_BUF_N    (BVP_BUF_SECONDS * FE_BVP_FS)
#define EDA_BUF_N    (EDA_BUF_SECONDS * FE_EDA_FS)

/* ── Shared buffers (PSRAM-allocated) ────────────────────────────────────── */
static float* s_acc_buf  = nullptr;
static float* s_bvp_buf  = nullptr;
static float* s_eda_buf  = nullptr;
static float* s_temp_buf = nullptr;

/* Write indices (updated by sensor_task) */
static volatile int s_acc_wr  = 0;
static volatile int s_bvp_wr  = 0;
static volatile int s_eda_wr  = 0;

/* Simple mutex via task notification is sufficient here because we only
   have one writer and one reader. A critical section around the index
   read is enough on Xtensa (32-bit aligned writes are atomic). */

static int s_sample_counter = 0;
static void get_sensor_data(float* ax, float* ay, float* az,
                             float* bvp,
                             float* eda, float* temp,
                             bool*  eda_valid)
{
    // ACC at 32 Hz
    mpu9250_read_acc(ax, ay, az);

    // BVP at 64 Hz — MAX30102 runs at 64 Hz internally;
    // we call twice per 32 Hz sensor_task tick to approximate 64 Hz.
    // For true 64 Hz, switch sensor_task to 15.6 ms period or use interrupt.
    *bvp = max30102_read_bvp();

    // EDA and temperature at 4 Hz (every 8th ACC sample = 250 ms)
    *eda_valid = (s_sample_counter % 8 == 0);
    if (*eda_valid) {
        *eda  = gsr_eda_read_us();
        *temp = max30102_read_temperature();  // die temp as skin-temp proxy
    }
    s_sample_counter++;
}

/* ── Sensor task (Core 0, high priority) ─────────────────────────────────── */
static void sensor_task(void* arg)
{
    const TickType_t period = pdMS_TO_TICKS(1000 / FE_ACC_FS);   /* 31 ms */
    TickType_t last_wake = xTaskGetTickCount();

    while (true) {
        float ax, ay, az, bvp, eda, temp;
        bool eda_valid;
        get_sensor_data(&ax, &ay, &az, &bvp, &eda, &temp, &eda_valid);

        /* Write ACC (circular) */
        int ai = (s_acc_wr % (ACC_BUF_SECONDS * FE_ACC_FS)) * 3;
        s_acc_buf[ai + 0] = ax;
        s_acc_buf[ai + 1] = ay;
        s_acc_buf[ai + 2] = az;
        s_acc_wr = s_acc_wr + 1;

        /* Write BVP (two samples per ACC tick at 64 Hz) */
        int bi = s_bvp_wr % (BVP_BUF_SECONDS * FE_BVP_FS);
        s_bvp_buf[bi]     = bvp;
        s_bvp_buf[bi + 1] = bvp;   /* duplicate until real 64 Hz driver */
        s_bvp_wr += 2;

        /* Write EDA/TEMP at 4 Hz */
        if (eda_valid) {
            int ei = s_eda_wr % (EDA_BUF_SECONDS * FE_EDA_FS);
            s_eda_buf[ei]  = eda;
            s_temp_buf[ei] = temp;
            s_eda_wr = s_eda_wr + 1;

        }

        vTaskDelayUntil(&last_wake, period);
    }
}

/* ── Analysis task (Core 0, normal priority) ─────────────────────────────── */
static FEFilterState s_fe_state;
static float s_features[FE_N_FEATURES];
static float s_zscore[BS_N_FEATURES];

/* Sleep tracking: accumulate epochs for night summary */
#define MAX_NIGHT_EPOCHS  1200   /* 10 h at 30 s per epoch */
static HEEpochResult s_night_epochs[MAX_NIGHT_EPOCHS];
static int           s_night_epoch_n = 0;

static void analysis_task(void* arg)
{
    const TickType_t period_60s = pdMS_TO_TICKS(60000);
    TickType_t last_wake = xTaskGetTickCount();

    int epoch_count = 0;   /* 60-s windows processed */

    while (true) {
        vTaskDelayUntil(&last_wake, period_60s);

        /* ── Snapshot 60 s of sensor data ─────────────────────────────── */
        static float acc_snap [FE_ACC_WINDOW_N * 3];
        static float bvp_snap [FE_BVP_WINDOW_N];
        static float eda_snap [FE_EDA_WINDOW_N];
        static float temp_snap[FE_EDA_WINDOW_N];

        /* Copy last 60 s from circular buffers */
        int acc_start = (s_acc_wr - FE_ACC_WINDOW_N + ACC_BUF_SECONDS * FE_ACC_FS)
                        % (ACC_BUF_SECONDS * FE_ACC_FS);
        for (int i = 0; i < FE_ACC_WINDOW_N; i++) {
            int idx = ((acc_start + i) % (ACC_BUF_SECONDS * FE_ACC_FS)) * 3;
            acc_snap[i * 3 + 0] = s_acc_buf[idx + 0];
            acc_snap[i * 3 + 1] = s_acc_buf[idx + 1];
            acc_snap[i * 3 + 2] = s_acc_buf[idx + 2];
        }

        int bvp_start = (s_bvp_wr - FE_BVP_WINDOW_N + BVP_BUF_SECONDS * FE_BVP_FS)
                        % (BVP_BUF_SECONDS * FE_BVP_FS);
        for (int i = 0; i < FE_BVP_WINDOW_N; i++) {
            bvp_snap[i] = s_bvp_buf[(bvp_start + i) % (BVP_BUF_SECONDS * FE_BVP_FS)];
        }

        int eda_start = (s_eda_wr - FE_EDA_WINDOW_N + EDA_BUF_SECONDS * FE_EDA_FS)
                        % (EDA_BUF_SECONDS * FE_EDA_FS);
        for (int i = 0; i < FE_EDA_WINDOW_N; i++) {
            int idx = (eda_start + i) % (EDA_BUF_SECONDS * FE_EDA_FS);
            eda_snap[i]  = s_eda_buf[idx];
            temp_snap[i] = s_temp_buf[idx];
        }

        /* ── Feature extraction ────────────────────────────────────────── */
        fe_extract_window(acc_snap, bvp_snap, eda_snap, temp_snap,
                          &s_fe_state, s_features);

        /* ── Personal baseline update ──────────────────────────────────── */
        bs_update(s_features);
        bs_zscore(s_features, s_zscore);

        int64_t now_s = (int64_t)(esp_timer_get_time() / 1000000LL);

        /* ── Head C: autonomic autoencoder ────────────────────────────── */
        HCResult hc = hc_infer(s_features);
        if (hc.valid && hc.is_anomalous) {
            ESP_LOGW(TAG, "Head C: anomaly score=%.4f (thresh=%.4f)",
                     hc.anomaly_score, hc.threshold);
            el_append(EL_TYPE_AUTONOMIC_ANOMALY, 1, 0,
                      EL_HEAD_C, s_features,
                      hc.anomaly_score, s_zscore[7], now_s);
        }

        /* ── Head D: stress classifier ────────────────────────────────── */
        HDResult hd = hd_infer(s_features);
        if (hd.valid && hd.predicted_class == 1 &&
            hd.prob[1] > 0.70f) {
            ESP_LOGI(TAG, "Head D: stress prob=%.3f", hd.prob[1]);
            el_append(EL_TYPE_STRESS_DETECTED, 1, 0,
                      EL_HEAD_D, s_features,
                      hd.prob[1], 0.0f, now_s);
        }

        /* ── Head B: tremor (every 4 s sub-window, run 15x per minute) ── */
        /* For now, classify the last 4 s of the ACC window */
        const float* acc_4s = acc_snap + (FE_ACC_WINDOW_N - HB_N_FFT) * 3;
        HBResult hb = hb_classify(acc_4s);
        if (hb.label == 1 && hb.sustained) {
            ESP_LOGW(TAG, "Head B: PD tremor sustained, freq=%.1f Hz",
                     hb.dom_freq_hz);
            el_append(EL_TYPE_PD_TREMOR, 2,
                      EL_FLAG_SUSTAINED, EL_HEAD_B,
                      s_features, 0.0f, hb.dom_freq_hz, now_s);
        } else if (hb.label == 2 && hb.sustained) {
            ESP_LOGW(TAG, "Head B: ET tremor sustained, freq=%.1f Hz",
                     hb.dom_freq_hz);
            el_append(EL_TYPE_ET_TREMOR, 2,
                      EL_FLAG_SUSTAINED, EL_HEAD_B,
                      s_features, 0.0f, hb.dom_freq_hz, now_s);
        }

        /* ── Head E: sleep epoch (every 30 s = every other 60-s window) ─ */
        if (epoch_count % 2 == 0) {
            /* Use first 30 s of ACC snapshot */
            HEEpochResult he = he_classify_epoch(acc_snap);
            if (s_night_epoch_n < MAX_NIGHT_EPOCHS) {
                s_night_epochs[s_night_epoch_n++] = he;
            }
        }

        /* ── Head E: nightly summary (once per day at ~06:00) ────────── */
        /* Simple proxy: every 1440 windows (24 h) produce a night summary */
        if (epoch_count > 0 && epoch_count % 1440 == 0) {
            HENightSummary night = he_summarise_night(
                s_night_epochs, s_night_epoch_n, 21.0f);
            s_night_epoch_n = 0;   /* reset for next night */

            if (night.valid) {
                if (night.efficiency < 0.75f || night.n_awakenings > 5) {
                    el_append(EL_TYPE_SLEEP_DISRUPTED, 1, 0,
                              EL_HEAD_E, s_features,
                              night.efficiency, (float)night.n_awakenings,
                              now_s);
                }

                /* ── Head F: circadian phase ───────────────────────────── */
                static bool sw_minutes[1440];
                memset(sw_minutes, 0, sizeof(sw_minutes));
                /* Build sleep/wake array from night epochs */
                for (int i = 0; i < s_night_epoch_n && i < 1440; i++) {
                    for (int m = 0; m < 2 && (i * 2 + m) < 1440; m++) {
                        sw_minutes[i * 2 + m] = s_night_epochs[i].is_sleep;
                    }
                }
                HFNightResult hf = hf_update(sw_minutes, night.midsleep_h);
                if (hf.valid && fabsf(hf.phase_z) > 2.0f) {
                    ELEventType ctype = (hf.phase_advance < 0)
                                        ? EL_TYPE_CIRCADIAN_DELAYED
                                        : EL_TYPE_CIRCADIAN_ADVANCED;
                    el_append(ctype, 1, 0, EL_HEAD_F,
                              s_features, hf.phase_z,
                              hf.phase_advance, now_s);
                }
            }

            /* Save baseline once per day */
            bs_save();
        }

        epoch_count++;

        /* Log status every 10 minutes */
        if (epoch_count % 10 == 0) {
            ELStatus st = el_status();
            ESP_LOGI(TAG, "Status: windows=%d records=%lu free=%lu B",
                     epoch_count,
                     (unsigned long)st.total_records,
                     (unsigned long)st.bytes_free);
        }
    }
}

/* ── app_main ────────────────────────────────────────────────────────────── */
extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "IDP Firmware starting. IDF %s", IDF_VER);

    /* NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    /* PSRAM buffers */
    s_acc_buf  = (float*)heap_caps_calloc(ACC_BUF_N,
                     sizeof(float), MALLOC_CAP_SPIRAM);
    s_bvp_buf  = (float*)heap_caps_calloc(BVP_BUF_N,
                     sizeof(float), MALLOC_CAP_SPIRAM);
    s_eda_buf  = (float*)heap_caps_calloc(EDA_BUF_N,
                     sizeof(float), MALLOC_CAP_SPIRAM);
    s_temp_buf = (float*)heap_caps_calloc(EDA_BUF_N,
                     sizeof(float), MALLOC_CAP_SPIRAM);

    if (!s_acc_buf || !s_bvp_buf || !s_eda_buf || !s_temp_buf) {
        ESP_LOGE(TAG, "PSRAM allocation failed");
        return;
    }
    ESP_LOGI(TAG, "PSRAM buffers allocated");

    /* Sensor initialisation */
    ESP_ERROR_CHECK(i2c_bus_init());
    ESP_ERROR_CHECK(max30102_init());
    ESP_ERROR_CHECK(mpu9250_init());
    ESP_ERROR_CHECK(gsr_eda_init());

    /* DSP + model initialisation */
    fft_init();
    fe_init(&s_fe_state);
    hb_init();
    he_init();
    hf_init();

    /* Storage */
    el_init();
    bs_init();

    /* ML heads — loaded from SPIFFS models partition */
    if (hc_init() != 0) ESP_LOGW(TAG, "Head C init failed (no model file yet)");
    if (hd_init() != 0) ESP_LOGW(TAG, "Head D init failed (no model file yet)");

    /* BLE (initialise but don't advertise yet) */
    ble_export_init();

    ESP_LOGI(TAG, "All subsystems initialised");

    /* Spawn tasks */
    xTaskCreatePinnedToCore(sensor_task,   "sensor",   4096,
                            NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(analysis_task, "analysis", 16384,
                            NULL, 3, NULL, 0);

    ESP_LOGI(TAG, "Tasks running");
}