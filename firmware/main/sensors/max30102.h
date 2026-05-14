#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * MAX30102 PPG sensor driver.
 * Interface: I2C, address 0x57.
 * Configured for heart rate mode (IR LED only), 100 Hz internal sample rate,
 * 18-bit ADC resolution, 411 us LED pulse width.
 *
 * BVP output: raw 18-bit IR ADC count (0–262143).
 * The feature extractor's Elgendi peak detector is amplitude-independent
 * so no voltage scaling is required before passing to max30102_read_bvp().
 *
 * Temperature: internal die temperature (±1°C), used as skin-temp proxy.
 */

// Initialise MAX30102. Returns ESP_OK on success, ESP_ERR_NOT_FOUND if
// the chip is absent or returns wrong PART_ID.
esp_err_t max30102_init(void);

// Read one BVP sample (IR channel, 18-bit count). Call at 64 Hz.
// Returns last valid sample if no new FIFO data is available.
float max30102_read_bvp(void);

// Read die temperature in degC. Blocks ~30 ms for conversion.
// Call at 4 Hz from sensor_task.
float max30102_read_temperature(void);

// True if the FIFO contains at least one unread sample.
bool max30102_data_ready(void);

#ifdef __cplusplus
}
#endif
