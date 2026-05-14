#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_err.h"
#include "esp_log.h"
#include "sensor_pins.h"
#include "i2c_bus.h"
#include "max30102.h"

static const char* TAG = "MAX30102";

// ── Register map ──────────────────────────────────────────────────────────
#define REG_INT_STATUS_1    0x00
#define REG_INT_ENABLE_1    0x02
#define REG_FIFO_WR_PTR     0x04
#define REG_FIFO_RD_PTR     0x06
#define REG_FIFO_DATA       0x07
#define REG_FIFO_CONFIG     0x08
#define REG_MODE_CONFIG     0x09
#define REG_SPO2_CONFIG     0x0A
#define REG_LED1_PA         0x0C
#define REG_LED2_PA         0x0D
#define REG_TEMP_INT        0x1F
#define REG_TEMP_FRAC       0x20
#define REG_TEMP_CONFIG     0x21
#define REG_REV_ID          0xFF
#define REG_PART_ID         0xFE

static float s_last_bvp = 0.0f;

esp_err_t max30102_init(void)
{
    // 1. Software reset
    i2c_write_byte(MAX30102_I2C_ADDR, REG_MODE_CONFIG, 0x40);
    vTaskDelay(pdMS_TO_TICKS(100));

    // 2. Verify PART_ID
    uint8_t part_id = i2c_read_byte(MAX30102_I2C_ADDR, REG_PART_ID);
    if (part_id != 0x15) {
        ESP_LOGE(TAG, "PART_ID mismatch: got 0x%02X, expected 0x15", part_id);
        return ESP_ERR_NOT_FOUND;
    }

    // 3. FIFO_CONFIG: no sample averaging, FIFO rollover enabled
    //    [7:5] SMP_AVE=000, [4] FIFO_ROLLOVER_EN=1, [3:0] FIFO_A_FULL=0
    i2c_write_byte(MAX30102_I2C_ADDR, REG_FIFO_CONFIG, 0x10);

    // 4. MODE_CONFIG: Heart rate mode (IR LED only)
    i2c_write_byte(MAX30102_I2C_ADDR, REG_MODE_CONFIG, 0x02);

    // 5. SPO2_CONFIG: ADC range=18-bit, SR=100 Hz, LED pulse width=411 us
    //    [6:5] ADC_RGE=11, [4:2] SR=001, [1:0] LED_PW=11  → 0x6F
    i2c_write_byte(MAX30102_I2C_ADDR, REG_SPO2_CONFIG, 0x6F);

    // 6. LED pulse amplitude: 0x24 = 7.2 mA
    i2c_write_byte(MAX30102_I2C_ADDR, REG_LED1_PA, 0x24);
    i2c_write_byte(MAX30102_I2C_ADDR, REG_LED2_PA, 0x24);

    // 7. Enable FIFO interrupts
    i2c_write_byte(MAX30102_I2C_ADDR, REG_INT_ENABLE_1, 0x60);

    // 8. Clear FIFO pointers
    i2c_write_byte(MAX30102_I2C_ADDR, REG_FIFO_WR_PTR, 0x00);
    i2c_write_byte(MAX30102_I2C_ADDR, REG_FIFO_RD_PTR, 0x00);

    ESP_LOGI(TAG, "MAX30102 ready (PART_ID=0x%02X, HR mode, IR only)", part_id);
    return ESP_OK;
}

float max30102_read_bvp(void)
{
    uint8_t wr_ptr = i2c_read_byte(MAX30102_I2C_ADDR, REG_FIFO_WR_PTR);
    uint8_t rd_ptr = i2c_read_byte(MAX30102_I2C_ADDR, REG_FIFO_RD_PTR);

    if (wr_ptr == rd_ptr) {
        return s_last_bvp;   /* no new data — return last valid sample */
    }

    /* Read 3 bytes (HR mode: one LED channel = 3 bytes per sample) */
    uint8_t raw[3] = {0};
    i2c_read_reg(MAX30102_I2C_ADDR, REG_FIFO_DATA, raw, 3);

    /* Combine to 18-bit sample: top 2 bits of raw[0] are valid */
    uint32_t sample = ((uint32_t)(raw[0] & 0x03) << 16)
                    | ((uint32_t)raw[1] << 8)
                    |  (uint32_t)raw[2];

    /* Advance read pointer (5-bit circular, wraps at 32) */
    rd_ptr = (rd_ptr + 1) & 0x1F;
    i2c_write_byte(MAX30102_I2C_ADDR, REG_FIFO_RD_PTR, rd_ptr);

    s_last_bvp = (float)sample;
    return s_last_bvp;
}

float max30102_read_temperature(void)
{
    /* Trigger one-shot temperature conversion */
    i2c_write_byte(MAX30102_I2C_ADDR, REG_TEMP_CONFIG, 0x01);
    vTaskDelay(pdMS_TO_TICKS(30));   /* datasheet: conversion ~29 ms */

    int8_t  temp_int  = (int8_t) i2c_read_byte(MAX30102_I2C_ADDR, REG_TEMP_INT);
    uint8_t temp_frac =           i2c_read_byte(MAX30102_I2C_ADDR, REG_TEMP_FRAC);

    /* Result in degC: integer + fractional (0.0625 °C steps) */
    return (float)temp_int + (float)temp_frac * 0.0625f;
}

bool max30102_data_ready(void)
{
    uint8_t wr = i2c_read_byte(MAX30102_I2C_ADDR, REG_FIFO_WR_PTR);
    uint8_t rd = i2c_read_byte(MAX30102_I2C_ADDR, REG_FIFO_RD_PTR);
    return wr != rd;
}
