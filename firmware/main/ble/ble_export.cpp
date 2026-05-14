#include "ble_export.h"
#include "event_logger.h"
#include "baseline_store.h"
#include "esp_log.h"
#include "esp_nimble_hci.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include <string.h>
#include <stdio.h>

static const char* TAG = "BLE";

/* ── UUIDs (custom 128-bit) ──────────────────────────────────────────────── */
/* Service: IDP Export Service */
static const ble_uuid128_t kSvcUUID = BLE_UUID128_INIT(
    0x49, 0x44, 0x50, 0x53, 0x56, 0x43, 0x00, 0x00,
    0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB);

/* Characteristic: STATUS (read) */
static const ble_uuid128_t kStatusUUID = BLE_UUID128_INIT(
    0x49, 0x44, 0x50, 0x53, 0x54, 0x41, 0x00, 0x00,
    0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB);

/* Characteristic: RECORDS (notify) */
static const ble_uuid128_t kRecordsUUID = BLE_UUID128_INIT(
    0x49, 0x44, 0x50, 0x52, 0x45, 0x43, 0x00, 0x00,
    0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB);

/* Characteristic: CONTROL (write) */
static const ble_uuid128_t kControlUUID = BLE_UUID128_INIT(
    0x49, 0x44, 0x50, 0x43, 0x54, 0x52, 0x00, 0x00,
    0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB);

/* ── State ───────────────────────────────────────────────────────────────── */
static uint16_t s_conn_handle   = BLE_HS_CONN_HANDLE_NONE;
static uint16_t s_records_handle = 0;
static bool     s_exporting     = false;
static bool     s_initialised   = false;

/* ── GATT characteristic access callbacks ────────────────────────────────── */
static int status_cb(uint16_t conn_handle, uint16_t attr_handle,
                     struct ble_gatt_access_ctxt* ctxt, void* arg)
{
    if (ctxt->op != BLE_GATT_ACCESS_OP_READ_CHR) return 0;
    ELStatus st = el_status();
    BSData   bs = bs_get();
    char buf[200];
    int n = snprintf(buf, sizeof(buf),
        "{\"records\":%lu,\"bytes_used\":%lu,\"bytes_free\":%lu,"
        "\"baseline_valid\":%s,\"calib_days\":%lu}",
        (unsigned long)st.total_records,
        (unsigned long)st.bytes_used,
        (unsigned long)st.bytes_free,
        bs.is_valid ? "true" : "false",
        (unsigned long)bs.calibration_days);
    os_mbuf_append(ctxt->om, buf, n);
    return 0;
}

static int records_cb(uint16_t conn_handle, uint16_t attr_handle,
                      struct ble_gatt_access_ctxt* ctxt, void* arg)
{
    /* Notify-only characteristic; reads return empty */
    return 0;
}

static int control_cb(uint16_t conn_handle, uint16_t attr_handle,
                      struct ble_gatt_access_ctxt* ctxt, void* arg)
{
    if (ctxt->op != BLE_GATT_ACCESS_OP_WRITE_CHR) return 0;
    uint8_t cmd = 0;
    uint16_t len = OS_MBUF_PKTLEN(ctxt->om);
    if (len < 1) return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    ble_hs_mbuf_to_flat(ctxt->om, &cmd, 1, NULL);

    switch (cmd) {
        case BLE_CMD_START_EXPORT:
            ESP_LOGI(TAG, "Export start requested");
            s_exporting = true;
            /* Stream records via notifications — done in export task */
            break;
        case BLE_CMD_STOP:
            s_exporting = false;
            ESP_LOGI(TAG, "Export stopped by client");
            break;
        case BLE_CMD_ERASE_LOG:
            ESP_LOGW(TAG, "Erase log command received");
            el_erase();
            break;
        default:
            return BLE_ATT_ERR_UNLIKELY;
    }
    return 0;
}

/* ── GATT service table ──────────────────────────────────────────────────── */
static const struct ble_gatt_svc_def kGattSvcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &kSvcUUID.u,
        .includes = NULL,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid       = &kStatusUUID.u,
                .access_cb  = status_cb,
                .arg        = NULL,
                .descriptors = NULL,
                .flags      = BLE_GATT_CHR_F_READ,
                .min_key_size = 0,
                .val_handle = NULL,
            },
            {
                .uuid       = &kRecordsUUID.u,
                .access_cb  = records_cb,
                .arg        = NULL,
                .descriptors = NULL,
                .flags      = BLE_GATT_CHR_F_NOTIFY,
                .min_key_size = 0,
                .val_handle = &s_records_handle,
            },
            {
                .uuid       = &kControlUUID.u,
                .access_cb  = control_cb,
                .arg        = NULL,
                .descriptors = NULL,
                .flags      = BLE_GATT_CHR_F_WRITE,
                .min_key_size = 0,
                .val_handle = NULL,
            },
            { 0 }
        },
    },
    { 0 }
};

/* ── GAP event handler ───────────────────────────────────────────────────── */
static int gap_event_cb(struct ble_gap_event* event, void* arg)
{
    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                s_conn_handle = event->connect.conn_handle;
                ESP_LOGI(TAG, "Client connected");
            }
            break;
        case BLE_GAP_EVENT_DISCONNECT:
            s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
            s_exporting   = false;
            ESP_LOGI(TAG, "Client disconnected");
            /* Restart advertising */
            ble_export_start();
            break;
        default:
            break;
    }
    return 0;
}

/* ── Advertising ─────────────────────────────────────────────────────────── */
static void start_advertising(void)
{
    struct ble_gap_adv_params adv_params = {};
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

    struct ble_hs_adv_fields fields = {};
    fields.flags                = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name                 = (const uint8_t*)BLE_DEVICE_NAME;
    fields.name_len             = strlen(BLE_DEVICE_NAME);
    fields.name_is_complete     = 1;

    ble_gap_adv_set_fields(&fields);
    ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER,
                      &adv_params, gap_event_cb, NULL);
    ESP_LOGI(TAG, "Advertising as '%s'", BLE_DEVICE_NAME);
}

static void on_sync(void)  { start_advertising(); }
static void on_reset(int r){ ESP_LOGW(TAG, "BLE reset: %d", r); }

static void nimble_host_task(void* arg)
{
    nimble_port_run();
    nimble_port_freertos_deinit();
}

/* ── Public API ──────────────────────────────────────────────────────────── */
int ble_export_init(void)
{
    if (s_initialised) return 0;
    nimble_port_init();
    ble_hs_cfg.sync_cb  = on_sync;
    ble_hs_cfg.reset_cb = on_reset;
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_gatts_count_cfg(kGattSvcs);
    ble_gatts_add_svcs(kGattSvcs);
    ble_svc_gap_device_name_set(BLE_DEVICE_NAME);
    s_initialised = true;
    ESP_LOGI(TAG, "NimBLE initialised");
    return 0;
}

int ble_export_start(void)
{
    if (!s_initialised) ble_export_init();
    nimble_port_freertos_init(nimble_host_task);
    return 0;
}

void ble_export_stop(void)
{
    if (s_conn_handle != BLE_HS_CONN_HANDLE_NONE) {
        ble_gap_terminate(s_conn_handle, BLE_ERR_REM_USER_CONN_TERM);
    }
    ble_gap_adv_stop();
}

bool ble_export_connected(void)
{
    return s_conn_handle != BLE_HS_CONN_HANDLE_NONE;
}