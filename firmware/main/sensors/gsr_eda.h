#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CJMCU-6701 GSR/EDA sensor driver.
 * Interface: Analog output → ESP32-S3 ADC1 (GPIO1, channel 0).
 *
 * The CJMCU-6701 outputs a voltage proportional to skin conductance.
 * Raw voltage is returned as a conductance proxy in arbitrary units.
 *
 * To calibrate:
 *   1. Measure skin conductance with a reference meter.
 *   2. Adjust GSR_SCALE_FACTOR in gsr_eda.cpp until values match.
 *   Typical resting SCL: 1–20 µS. Stress response: 2–10× increase.
 *
 * The feature extractor uses SCL for tonic EDA and SCR count for phasic EDA.
 * Absolute calibration improves interpretability but is not required for
 * the anomaly detector (which z-scores against personal baseline).
 */

// Initialise ADC1 with calibration (curve fitting preferred, line fitting fallback).
esp_err_t gsr_eda_init(void);

// Read one EDA sample. Returns conductance proxy (voltage × GSR_SCALE_FACTOR).
// Call at 4 Hz from sensor_task.
float gsr_eda_read_us(void);

#ifdef __cplusplus
}
#endif
