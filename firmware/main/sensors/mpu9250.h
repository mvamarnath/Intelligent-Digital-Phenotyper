#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * MPU9250 / MPU6500 accelerometer driver.
 * Interface: I2C, address 0x68 (AD0=GND) or 0x69 (AD0=VCC).
 * Configured for accelerometer-only use at ~32 Hz, ±2g full-scale range.
 * Gyroscope and magnetometer are disabled to save power.
 *
 * Output: accelerometer in g units (raw int16 / 16384).
 * feature_extractor.cpp expects g units directly — no further scaling needed.
 */

// Initialise MPU9250. Returns ESP_OK on success,
// ESP_ERR_NOT_FOUND if WHO_AM_I does not match.
esp_err_t mpu9250_init(void);

// Read one accelerometer sample. Fills ax, ay, az in g units.
// Call at 32 Hz from sensor_task.
void mpu9250_read_acc(float* ax, float* ay, float* az);

#ifdef __cplusplus
}
#endif
