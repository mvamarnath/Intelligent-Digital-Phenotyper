#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_log.h"
#include "sensor_pins.h"
#include "gsr_eda.h"

static const char* TAG = "GSR_EDA";

/* Calibration factor: voltage_v × GSR_SCALE_FACTOR → conductance proxy.
 * Calibrate against a known conductance reference for your specific board.
 * Default 1.0 returns voltage in volts as a conductance-proportional value. */
#define GSR_SCALE_FACTOR    1.0f

static adc_oneshot_unit_handle_t s_adc_handle  = NULL;
static adc_cali_handle_t         s_cali_handle = NULL;
static bool                      s_calibrated  = false;

esp_err_t gsr_eda_init(void)
{
    /* 1. Create ADC oneshot unit */
    adc_oneshot_unit_init_cfg_t unit_cfg = {};
    unit_cfg.unit_id  = GSR_ADC_UNIT;
    unit_cfg.clk_src  = ADC_RTC_CLK_SRC_DEFAULT;
    unit_cfg.ulp_mode = ADC_ULP_MODE_DISABLE;

    esp_err_t err = adc_oneshot_new_unit(&unit_cfg, &s_adc_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "adc_oneshot_new_unit failed: %s", esp_err_to_name(err));
        return err;
    }

    /* 2. Configure channel — ADC_ATTEN_DB_12 covers 0–3.3 V range */
    adc_oneshot_chan_cfg_t chan_cfg = {};
    chan_cfg.atten    = ADC_ATTEN_DB_12;   /* 0–3.3 V; DB_11 is deprecated alias */
    chan_cfg.bitwidth = ADC_BITWIDTH_12;

    err = adc_oneshot_config_channel(s_adc_handle, GSR_ADC_CHANNEL, &chan_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "adc_oneshot_config_channel failed: %s", esp_err_to_name(err));
        return err;
    }

    /* 3. Calibration — prefer curve fitting (ESP32-S3 native) */
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    {
        adc_cali_curve_fitting_config_t cali_cfg = {};
        cali_cfg.unit_id  = GSR_ADC_UNIT;
        cali_cfg.chan     = GSR_ADC_CHANNEL;
        cali_cfg.atten    = ADC_ATTEN_DB_12;
        cali_cfg.bitwidth = ADC_BITWIDTH_12;
        if (adc_cali_create_scheme_curve_fitting(&cali_cfg, &s_cali_handle) == ESP_OK) {
            s_calibrated = true;
            ESP_LOGI(TAG, "ADC calibration: curve fitting");
        }
    }
#endif

    /* 4. Fall back to line fitting if curve fitting is unavailable */
    if (!s_calibrated) {
#if ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED
        adc_cali_line_fitting_config_t lf_cfg = {};
        lf_cfg.unit_id  = GSR_ADC_UNIT;
        lf_cfg.atten    = ADC_ATTEN_DB_12;
        lf_cfg.bitwidth = ADC_BITWIDTH_12;
        if (adc_cali_create_scheme_line_fitting(&lf_cfg, &s_cali_handle) == ESP_OK) {
            s_calibrated = true;
            ESP_LOGI(TAG, "ADC calibration: line fitting");
        }
#endif
    }

    if (!s_calibrated) {
        ESP_LOGW(TAG, "ADC calibration unavailable — using raw linear approximation");
    }

    ESP_LOGI(TAG, "GSR/EDA ADC ready (GPIO%d, ADC1_CH%d, scale=%.2f)",
             PIN_GSR_ADC, (int)GSR_ADC_CHANNEL, GSR_SCALE_FACTOR);
    return ESP_OK;
}

float gsr_eda_read_us(void)
{
    int raw = 0;
    adc_oneshot_read(s_adc_handle, GSR_ADC_CHANNEL, &raw);

    float voltage_v;
    if (s_calibrated) {
        int mv = 0;
        adc_cali_raw_to_voltage(s_cali_handle, raw, &mv);
        voltage_v = (float)mv / 1000.0f;
    } else {
        /* Linear fallback: 0–4095 counts → 0–3.3 V */
        voltage_v = (float)raw * 3.3f / 4095.0f;
    }

    return voltage_v * GSR_SCALE_FACTOR;
}
