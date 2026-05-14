#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "event_logger.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * BLE Export — NimBLE GATT server for clinician packet transfer.
 *
 * Advertises as "IDP-Device" when export mode is triggered.
 * Provides three GATT characteristics:
 *
 *   STATUS  (read)       — device status JSON (battery, record count, etc.)
 *   RECORDS (notify)     — streams ELRecord structs one per notification
 *   CONTROL (write)      — accepts commands: START_EXPORT, STOP, ERASE
 *
 * Export flow:
 *   1. Clinician app connects and writes START_EXPORT to CONTROL.
 *   2. Device streams all ELRecord entries via RECORDS notifications.
 *   3. Device writes END_OF_RECORDS sentinel.
 *   4. Clinician app disconnects or writes STOP.
 *
 * BLE is disabled during normal operation to save power.
 * Call ble_export_start() to enable; it auto-disables after
 * BLE_EXPORT_TIMEOUT_S seconds of inactivity.
 */

#define BLE_DEVICE_NAME         "IDP-Device"
#define BLE_EXPORT_TIMEOUT_S    300      /* 5 minutes */
#define BLE_MTU                 512

/* Control commands (1 byte) */
#define BLE_CMD_START_EXPORT    0x01
#define BLE_CMD_STOP            0x02
#define BLE_CMD_ERASE_LOG       0xEE    /* requires double confirmation */

/**
 * Initialise NimBLE stack. Call once at startup (does not start advertising).
 */
int  ble_export_init(void);

/**
 * Enable BLE advertising and GATT server.
 * Call when user triggers export (e.g. button press or USB detection).
 */
int  ble_export_start(void);

/**
 * Disable BLE (call after export complete or timeout).
 */
void ble_export_stop(void);

/**
 * True if a client is currently connected.
 */
bool ble_export_connected(void);

#ifdef __cplusplus
}
#endif