#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_err.h"
#include "esp_log.h"
#include "sensor_pins.h"
#include "i2c_bus.h"
#include "mpu9250.h"

static const char* TAG = "MPU9250";

// ── Register map ──────────────────────────────────────────────────────────
#define REG_SMPLRT_DIV      0x19
#define REG_CONFIG          0x1A
#define REG_ACCEL_CONFIG    0x1C
#define REG_ACCEL_CONFIG2   0x1D
#define REG_ACCEL_XOUT_H    0x3B
#define REG_PWR_MGMT_1      0x6B
#define REG_PWR_MGMT_2      0x6C
#define REG_WHO_AM_I        0x75

#define WHO_AM_I_MPU9250    0x71
#define WHO_AM_I_MPU6500    0x70

esp_err_t mpu9250_init(void)
{
    // 1. Reset device (bit 7 of PWR_MGMT_1)
    i2c_write_byte(MPU9250_I2C_ADDR, REG_PWR_MGMT_1, 0x80);
    vTaskDelay(pdMS_TO_TICKS(100));

    // 2. Wake up, use internal 20 MHz oscillator
    i2c_write_byte(MPU9250_I2C_ADDR, REG_PWR_MGMT_1, 0x00);
    vTaskDelay(pdMS_TO_TICKS(10));

    // 3. Verify WHO_AM_I (accept both MPU9250 and MPU6500)
    uint8_t who = i2c_read_byte(MPU9250_I2C_ADDR, REG_WHO_AM_I);
    if (who != WHO_AM_I_MPU9250 && who != WHO_AM_I_MPU6500) {
        ESP_LOGE(TAG, "WHO_AM_I mismatch: 0x%02X (expected 0x71 or 0x70)", who);
        return ESP_ERR_NOT_FOUND;
    }

    // 4. Disable gyroscope axes to save power
    //    PWR_MGMT_2 bits[2:0]=1 → disable X/Y/Z gyro
    i2c_write_byte(MPU9250_I2C_ADDR, REG_PWR_MGMT_2, 0x07);

    // 5. Sample rate: SMPLRT_DIV=31 → 1000 Hz / (1+31) ≈ 31.25 Hz
    i2c_write_byte(MPU9250_I2C_ADDR, REG_SMPLRT_DIV, 31);

    // 6. CONFIG: DLPF_CFG=3 → 41 Hz bandwidth (reduces noise)
    i2c_write_byte(MPU9250_I2C_ADDR, REG_CONFIG, 0x03);

    // 7. ACCEL_CONFIG: AFS_SEL=00 → ±2g full-scale range
    i2c_write_byte(MPU9250_I2C_ADDR, REG_ACCEL_CONFIG, 0x00);

    // 8. ACCEL_CONFIG2: A_DLPFCFG=3 → 41 Hz accel bandwidth
    i2c_write_byte(MPU9250_I2C_ADDR, REG_ACCEL_CONFIG2, 0x03);

    ESP_LOGI(TAG, "MPU9250 ready (WHO_AM_I=0x%02X, accel-only ±2g ~32 Hz)", who);
    return ESP_OK;
}

void mpu9250_read_acc(float* ax, float* ay, float* az)
{
    uint8_t buf[6] = {0};
    i2c_read_reg(MPU9250_I2C_ADDR, REG_ACCEL_XOUT_H, buf, 6);

    int16_t raw_x = (int16_t)((uint16_t)(buf[0] << 8) | buf[1]);
    int16_t raw_y = (int16_t)((uint16_t)(buf[2] << 8) | buf[3]);
    int16_t raw_z = (int16_t)((uint16_t)(buf[4] << 8) | buf[5]);

    /* ±2g full-scale → 16384 LSB/g */
    *ax = (float)raw_x / 16384.0f;
    *ay = (float)raw_y / 16384.0f;
    *az = (float)raw_z / 16384.0f;
}
