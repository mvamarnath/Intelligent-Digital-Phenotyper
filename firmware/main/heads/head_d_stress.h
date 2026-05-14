#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Head D: Stress / Affect Classifier
 *
 * 3-class softmax output:
 *   class 0 = baseline
 *   class 1 = stress/arousal
 *   class 2 = other (amusement / meditation)
 */

#define HD_N_FEATURES   10
#define HD_N_CLASSES     3
#define HD_MODEL_PATH   "/models/stress_clf_int8.tflite"

typedef struct {
    float  prob[HD_N_CLASSES];  /* softmax probabilities */
    int    predicted_class;     /* argmax */
    bool   valid;
} HDResult;

int      hd_init(void);
HDResult hd_infer(const float* features);

#ifdef __cplusplus
}
#endif