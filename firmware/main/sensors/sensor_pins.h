// main/sensors/sensor_pins.h
// Single source of truth for all GPIO assignments.
// Change pin numbers here only — every driver includes this file.
// All pins are plain integers; drivers cast to gpio_num_t at point of use.
#pragma once

// ── I2C bus (shared by MAX30102 and MPU9250) ─────────────────────────────
#define PIN_I2C_SDA         8       // GPIO8
#define PIN_I2C_SCL         9       // GPIO9
#define I2C_PORT            I2C_NUM_0
#define I2C_FREQ_HZ         400000  // 400 kHz fast mode

// ── MAX30102 PPG sensor ──────────────────────────────────────────────────
#define MAX30102_I2C_ADDR   0x57
#define PIN_MAX30102_INT    7       // GPIO7 — data-ready interrupt (active low)

// ── MPU9250 IMU ─────────────────────────────────────────────────────────
#define MPU9250_I2C_ADDR    0x68    // AD0=GND; use 0x69 if AD0=VCC
#define PIN_MPU9250_INT     6       // GPIO6 — data-ready interrupt

// ── CJMCU-6701 GSR/EDA sensor (analog output) ───────────────────────────
#define PIN_GSR_ADC         1       // GPIO1 — ADC1_CH0
#define GSR_ADC_CHANNEL     ADC_CHANNEL_0
#define GSR_ADC_UNIT        ADC_UNIT_1
#define GSR_ADC_ATTEN       ADC_ATTEN_DB_12  // 0–3.3 V range

// ── NTC skin temperature (optional) ─────────────────────────────────────
// Uncomment if an external NTC thermistor is fitted on GPIO2.
// Otherwise MAX30102 die temperature is used as a proxy.
// #define PIN_TEMP_NTC_ADC  2      // GPIO2 — ADC1_CH1

// ── Export trigger button ─────────────────────────────────────────────────
#define PIN_EXPORT_BTN      0       // GPIO0 — BOOT button on DevKitC-1
