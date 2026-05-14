#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include <string.h>
#include <assert.h>
#include "sensor_pins.h"
#include "i2c_bus.h"

static const char* TAG = "I2C_BUS";

#define I2C_TIMEOUT_MS  100
#define MAX_DEVICES     4
#define MAX_WRITE_LEN   64

typedef struct {
    uint8_t                 addr;
    i2c_master_dev_handle_t handle;
} dev_entry_t;

static i2c_master_bus_handle_t s_bus   = NULL;
static SemaphoreHandle_t       s_mutex = NULL;
static bool                    s_init  = false;
static dev_entry_t             s_devs[MAX_DEVICES];
static int                     s_dev_n = 0;

/* Returns (or lazily creates) the device handle for addr.
   Must be called with s_mutex held. */
static i2c_master_dev_handle_t get_dev(uint8_t addr)
{
    for (int i = 0; i < s_dev_n; i++) {
        if (s_devs[i].addr == addr) return s_devs[i].handle;
    }
    if (s_dev_n >= MAX_DEVICES) {
        ESP_LOGE(TAG, "Device table full");
        return NULL;
    }

    i2c_device_config_t cfg = {};
    cfg.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    cfg.device_address  = addr;
    cfg.scl_speed_hz    = I2C_FREQ_HZ;
    cfg.scl_wait_us     = 0;
    /* flags.disable_ack_check = 0 (default — check ACK) */

    i2c_master_dev_handle_t hdl = NULL;
    if (i2c_master_bus_add_device(s_bus, &cfg, &hdl) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add I2C device 0x%02X", addr);
        return NULL;
    }
    s_devs[s_dev_n].addr   = addr;
    s_devs[s_dev_n].handle = hdl;
    s_dev_n++;
    return hdl;
}

esp_err_t i2c_bus_init(void)
{
    if (s_init) return ESP_OK;

    s_mutex = xSemaphoreCreateMutex();
    if (!s_mutex) return ESP_ERR_NO_MEM;

    i2c_master_bus_config_t cfg = {};
    cfg.i2c_port          = I2C_PORT;
    cfg.sda_io_num        = (gpio_num_t)PIN_I2C_SDA;
    cfg.scl_io_num        = (gpio_num_t)PIN_I2C_SCL;
    cfg.clk_source        = I2C_CLK_SRC_DEFAULT;
    cfg.glitch_ignore_cnt = 7;
    cfg.intr_priority     = 0;
    cfg.trans_queue_depth = 0;
    cfg.flags.enable_internal_pullup = false;   /* use external 4.7 k pull-ups */

    esp_err_t err = i2c_new_master_bus(&cfg, &s_bus);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2c_new_master_bus failed: %s", esp_err_to_name(err));
        vSemaphoreDelete(s_mutex);
        s_mutex = NULL;
        return err;
    }

    s_init = true;
    ESP_LOGI(TAG, "I2C ready (SDA=GPIO%d SCL=GPIO%d %d kHz)",
             PIN_I2C_SDA, PIN_I2C_SCL, I2C_FREQ_HZ / 1000);
    return ESP_OK;
}

esp_err_t i2c_write_reg(uint8_t i2c_addr, uint8_t reg_addr,
                         const uint8_t* data, size_t len)
{
    assert(len < MAX_WRITE_LEN);
    uint8_t buf[MAX_WRITE_LEN + 1];
    buf[0] = reg_addr;
    memcpy(buf + 1, data, len);

    xSemaphoreTake(s_mutex, portMAX_DELAY);
    i2c_master_dev_handle_t dev = get_dev(i2c_addr);
    esp_err_t err = dev
        ? i2c_master_transmit(dev, buf, len + 1, I2C_TIMEOUT_MS)
        : ESP_ERR_NOT_FOUND;
    xSemaphoreGive(s_mutex);
    return err;
}

esp_err_t i2c_write_byte(uint8_t i2c_addr, uint8_t reg_addr, uint8_t value)
{
    uint8_t buf[2] = { reg_addr, value };

    xSemaphoreTake(s_mutex, portMAX_DELAY);
    i2c_master_dev_handle_t dev = get_dev(i2c_addr);
    esp_err_t err = dev
        ? i2c_master_transmit(dev, buf, 2, I2C_TIMEOUT_MS)
        : ESP_ERR_NOT_FOUND;
    xSemaphoreGive(s_mutex);
    return err;
}

esp_err_t i2c_read_reg(uint8_t i2c_addr, uint8_t reg_addr,
                        uint8_t* buf, size_t len)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    i2c_master_dev_handle_t dev = get_dev(i2c_addr);
    esp_err_t err = dev
        ? i2c_master_transmit_receive(dev, &reg_addr, 1,
                                      buf, len, I2C_TIMEOUT_MS)
        : ESP_ERR_NOT_FOUND;
    xSemaphoreGive(s_mutex);
    return err;
}

uint8_t i2c_read_byte(uint8_t i2c_addr, uint8_t reg_addr)
{
    uint8_t val = 0;
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    i2c_master_dev_handle_t dev = get_dev(i2c_addr);
    if (dev) {
        i2c_master_transmit_receive(dev, &reg_addr, 1,
                                    &val, 1, I2C_TIMEOUT_MS);
    }
    xSemaphoreGive(s_mutex);
    return val;
}
