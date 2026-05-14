#pragma once
#include "driver/i2c_master.h"   // IDF 5.x new I2C master API
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * I2C bus manager — shared between MAX30102 and MPU9250.
 * All functions are protected by an internal FreeRTOS mutex.
 * Safe to call from multiple tasks.
 */

// Initialise I2C master on I2C_PORT with SDA/SCL from sensor_pins.h.
// Idempotent — safe to call more than once.
esp_err_t i2c_bus_init(void);

// Write len bytes from data to reg_addr on device at i2c_addr.
esp_err_t i2c_write_reg(uint8_t i2c_addr, uint8_t reg_addr,
                         const uint8_t* data, size_t len);

// Write a single byte value to reg_addr.
esp_err_t i2c_write_byte(uint8_t i2c_addr, uint8_t reg_addr, uint8_t value);

// Read len bytes starting at reg_addr on device at i2c_addr into buf.
esp_err_t i2c_read_reg(uint8_t i2c_addr, uint8_t reg_addr,
                        uint8_t* buf, size_t len);

// Read a single byte from reg_addr. Returns 0 on error.
uint8_t i2c_read_byte(uint8_t i2c_addr, uint8_t reg_addr);

#ifdef __cplusplus
}
#endif
