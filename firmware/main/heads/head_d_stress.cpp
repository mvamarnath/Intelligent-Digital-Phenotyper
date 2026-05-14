#include "head_d_stress.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

static const char* TAG = "HeadD";

/* ── Preprocessing constants from stress_clf_quant_params.h ──────────────── */
static const float kImputerMedian[HD_N_FEATURES] = {
    0.004314f, 0.012491f, 2.750000f, 0.086349f,
    70.766251f, 71.159378f, 97.747726f,
    0.777245f, 0.000000f, 32.952003f
};
static const float kScalerCenter[HD_N_FEATURES] = {
    0.004314f, 0.012491f, 2.750000f, 0.086349f,
    70.766251f, 71.159378f, 97.747726f,
    0.777245f, 0.000000f, 32.952003f
};
static const float kScalerScale[HD_N_FEATURES] = {
    0.012688f, 0.026048f, 1.000000f, 1.000000f,
    10.250000f, 36.879562f, 53.203880f,
    2.171066f, 1.000000f, 2.140253f
};
/* Input/output quantisation from stress_clf_quant_params.h */
static const float  kInputScale  = 0.05588235f;
static const int8_t kInputZP     = -88;
static const float  kOutputScale = 0.00390625f;
static const int8_t kOutputZP    = -128;

/* ── TFLM objects ────────────────────────────────────────────────────────── */
#define HD_ARENA_SIZE   (8 * 1024)
static uint8_t  s_arena[HD_ARENA_SIZE];
static uint8_t* s_model_buf  = nullptr;
static size_t   s_model_size = 0;

static tflite::MicroMutableOpResolver<6> s_resolver;
static tflite::MicroInterpreter*         s_interp = nullptr;
static bool s_ready = false;

/* ── Model loading ───────────────────────────────────────────────────────── */
static int load_model(void)
{
    FILE* f = fopen(HD_MODEL_PATH, "rb");
    if (!f) { ESP_LOGE(TAG, "Cannot open %s", HD_MODEL_PATH); return -1; }
    fseek(f, 0, SEEK_END);
    s_model_size = (size_t)ftell(f);
    rewind(f);
    s_model_buf = (uint8_t*)heap_caps_malloc(
        s_model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_model_buf) s_model_buf = (uint8_t*)malloc(s_model_size);
    if (!s_model_buf) {
        ESP_LOGE(TAG, "OOM"); fclose(f); return -1;
    }
    fread(s_model_buf, 1, s_model_size, f);
    fclose(f);
    ESP_LOGI(TAG, "Model loaded: %u bytes", (unsigned)s_model_size);
    return 0;
}

/* ── Initialisation ──────────────────────────────────────────────────────── */
int hd_init(void)
{
    if (load_model() != 0) return -1;
    const tflite::Model* model = tflite::GetModel(s_model_buf);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch"); return -1;
    }

    s_resolver.AddFullyConnected();
    s_resolver.AddRelu();
    s_resolver.AddSoftmax();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();
    s_resolver.AddReshape();

    static tflite::MicroInterpreter interp(
        model, s_resolver, s_arena, HD_ARENA_SIZE);
    s_interp = &interp;

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed"); return -1;
    }
    s_ready = true;
    ESP_LOGI(TAG, "Ready");
    return 0;
}

/* ── Inference ───────────────────────────────────────────────────────────── */
HDResult hd_infer(const float* features)
{
    HDResult res = {{0}, 0, false};
    if (!s_ready) return res;

    TfLiteTensor* inp = s_interp->input(0);

    /* Preprocess: impute NaN + RobustScale + quantise */
    for (int i = 0; i < HD_N_FEATURES; i++) {
        float v = features[i];
        if (v != v) v = kImputerMedian[i];
        float sc = kScalerScale[i] > 0.0f ? kScalerScale[i] : 1.0f;
        v = (v - kScalerCenter[i]) / sc;
        int32_t q = (int32_t)roundf(v / kInputScale) + kInputZP;
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        inp->data.int8[i] = (int8_t)q;
    }

    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGW(TAG, "Invoke failed"); return res;
    }

    TfLiteTensor* out = s_interp->output(0);

    /* Dequantise outputs using the known params (not tensor params for INT8) */
    float best = -1e9f;
    for (int c = 0; c < HD_N_CLASSES; c++) {
        float p = ((float)out->data.int8[c] - kOutputZP) * kOutputScale;
        res.prob[c] = p;
        if (p > best) { best = p; res.predicted_class = c; }
    }
    res.valid = true;
    return res;
}