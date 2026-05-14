#include "head_c_autoencoder.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "nvs.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

static const char* TAG = "HeadC";

/* ── Preprocessing constants from autoencoder_quant_params.h ─────────────── */
static const float kImputerMedian[HC_N_FEATURES] = {
    0.005672f, 0.019278f, 2.250000f, 0.083659f,
    70.086624f, 65.159912f, 90.035645f,
    0.398046f, 0.000000f, 33.593834f
};
static const float kScalerCenter[HC_N_FEATURES] = {
    0.005672f, 0.019278f, 2.250000f, 0.083659f,
    70.086624f, 65.159912f, 90.035645f,
    0.398046f, 0.000000f, 33.593834f
};
static const float kScalerScale[HC_N_FEATURES] = {
    0.009504f, 0.020178f, 1.000000f, 1.000000f,
    9.363317f, 34.840607f, 50.483128f,
    1.249815f, 1.000000f, 1.431730f
};
static const float  kInputScale  = 0.05854225f;
static const int8_t kInputZP     = -78;
static const float  kOutputScale = 0.02394544f;
static const int8_t kOutputZP    = -38;

/* ── TFLM objects ────────────────────────────────────────────────────────── */
#define HC_ARENA_SIZE   (8 * 1024)
static uint8_t  s_arena[HC_ARENA_SIZE];
static uint8_t* s_model_buf  = nullptr;
static size_t   s_model_size = 0;

static tflite::MicroMutableOpResolver<4> s_resolver;
static tflite::MicroInterpreter*         s_interp = nullptr;
static bool s_ready = false;

/* ── NVS threshold ───────────────────────────────────────────────────────── */
static float s_threshold = HC_DEFAULT_THRESH;

static void load_threshold_nvs(void)
{
    nvs_handle_t h;
    if (nvs_open(HC_NVS_NS, NVS_READONLY, &h) != ESP_OK) return;
    uint32_t raw = 0;
    if (nvs_get_u32(h, HC_NVS_THRESH, &raw) == ESP_OK)
        memcpy(&s_threshold, &raw, sizeof(float));
    nvs_close(h);
}

/* ── Model loading ───────────────────────────────────────────────────────── */
static int load_model(void)
{
    FILE* f = fopen(HC_MODEL_PATH, "rb");
    if (!f) { ESP_LOGE(TAG, "Cannot open %s", HC_MODEL_PATH); return -1; }
    fseek(f, 0, SEEK_END);
    s_model_size = (size_t)ftell(f);
    rewind(f);
    s_model_buf = (uint8_t*)heap_caps_malloc(
        s_model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_model_buf) s_model_buf = (uint8_t*)malloc(s_model_size);
    if (!s_model_buf) {
        ESP_LOGE(TAG, "OOM (%u bytes)", (unsigned)s_model_size);
        fclose(f); return -1;
    }
    fread(s_model_buf, 1, s_model_size, f);
    fclose(f);
    ESP_LOGI(TAG, "Model loaded: %u bytes", (unsigned)s_model_size);
    return 0;
}

/* ── Initialisation ──────────────────────────────────────────────────────── */
int hc_init(void)
{
    if (load_model() != 0) return -1;

    const tflite::Model* model = tflite::GetModel(s_model_buf);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch"); return -1;
    }

    s_resolver.AddFullyConnected();
    s_resolver.AddRelu();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();

    static tflite::MicroInterpreter interp(
        model, s_resolver, s_arena, HC_ARENA_SIZE);
    s_interp = &interp;

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed"); return -1;
    }

    load_threshold_nvs();
    s_ready = true;
    ESP_LOGI(TAG, "Ready. Threshold=%.4f", s_threshold);
    return 0;
}

/* ── Preprocessing ───────────────────────────────────────────────────────── */
static void preprocess(const float* raw, int8_t* q_out)
{
    for (int i = 0; i < HC_N_FEATURES; i++) {
        float v = raw[i];
        if (v != v) v = kImputerMedian[i];   /* impute NaN */
        float sc = kScalerScale[i] > 0.0f ? kScalerScale[i] : 1.0f;
        v = (v - kScalerCenter[i]) / sc;
        int32_t q = (int32_t)roundf(v / kInputScale) + kInputZP;
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        q_out[i] = (int8_t)q;
    }
}

/* ── Inference ───────────────────────────────────────────────────────────── */
HCResult hc_infer(const float* features)
{
    HCResult res = {0.0f, false, s_threshold, false};
    if (!s_ready) return res;

    TfLiteTensor* inp = s_interp->input(0);
    preprocess(features, inp->data.int8);

    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGW(TAG, "Invoke failed"); return res;
    }

    TfLiteTensor* out = s_interp->output(0);

    /* Scaled input for MSE computation */
    float scaled_in[HC_N_FEATURES];
    for (int i = 0; i < HC_N_FEATURES; i++) {
        float v = features[i];
        if (v != v) v = kImputerMedian[i];
        float sc = kScalerScale[i] > 0.0f ? kScalerScale[i] : 1.0f;
        scaled_in[i] = (v - kScalerCenter[i]) / sc;
    }

    float mse = 0.0f;
    for (int i = 0; i < HC_N_FEATURES; i++) {
        float recon = ((float)out->data.int8[i] - kOutputZP) * kOutputScale;
        float diff  = scaled_in[i] - recon;
        mse += diff * diff;
    }
    mse /= HC_N_FEATURES;

    res.anomaly_score = mse;
    res.is_anomalous  = (mse > s_threshold);
    res.threshold     = s_threshold;
    res.valid         = true;
    return res;
}

/* ── Threshold management ────────────────────────────────────────────────── */
void hc_set_threshold(float threshold)
{
    s_threshold = threshold;
    nvs_handle_t h;
    if (nvs_open(HC_NVS_NS, NVS_READWRITE, &h) != ESP_OK) return;
    uint32_t raw = 0;
    memcpy(&raw, &threshold, sizeof(float));
    nvs_set_u32(h, HC_NVS_THRESH, raw);
    nvs_commit(h);
    nvs_close(h);
}

float hc_get_threshold(void) { return s_threshold; }