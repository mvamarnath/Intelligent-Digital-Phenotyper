#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Event Logger — append-only encrypted log on the eventlog SPIFFS partition.
 *
 * Each event is a fixed-size 128-byte record:
 *   [4]  magic       0xIDPE
 *   [4]  sequence    monotonic counter
 *   [8]  timestamp   unix epoch seconds (int64)
 *   [1]  event_type  EL_TYPE_* enum
 *   [1]  severity    0=info 1=warn 2=alert
 *   [2]  flags       bitmask
 *   [40] features    float32[10] — the raw feature vector at event time
 *   [4]  anomaly_score float32
 *   [4]  aux_float   head-specific value (tremor freq, phase_z, etc.)
 *   [1]  head_id     which head generated this event
 *   [59] reserved    zero-padded to 128 bytes
 *
 * Records are AES-128-CTR encrypted with a device key stored in NVS eFuse.
 * For the current implementation we use a placeholder XOR cipher that is
 * replaced with mbedTLS AES in the security hardening pass.
 *
 * The log is append-only. Deletion is only possible via full partition erase
 * (factory reset). This guarantees longitudinal integrity.
 */

#define EL_RECORD_SIZE      128
#define EL_MAGIC            0x49445045UL   /* "IDPE" */
#define EL_MOUNT_POINT      "/eventlog"
#define EL_FILE_PATH        "/eventlog/events.bin"
#define EL_NVS_NS           "eventlog"
#define EL_NVS_SEQ          "seq"

/* Event types */
typedef enum {
    EL_TYPE_AUTONOMIC_ANOMALY   = 0x01,
    EL_TYPE_STRESS_DETECTED     = 0x02,
    EL_TYPE_PD_TREMOR           = 0x10,
    EL_TYPE_ET_TREMOR           = 0x11,
    EL_TYPE_TREMOR_INDETERMINATE= 0x12,
    EL_TYPE_SEIZURE_LIKE        = 0x13,
    EL_TYPE_SLEEP_DISRUPTED     = 0x20,
    EL_TYPE_CIRCADIAN_DELAYED   = 0x21,
    EL_TYPE_CIRCADIAN_ADVANCED  = 0x22,
    EL_TYPE_BASELINE_UPDATED    = 0x30,
    EL_TYPE_SYSTEM              = 0xFF,
} ELEventType;

/* Head IDs */
#define EL_HEAD_C   0xC0   /* autoencoder */
#define EL_HEAD_D   0xD0   /* stress clf  */
#define EL_HEAD_B   0xB0   /* tremor      */
#define EL_HEAD_E   0xE0   /* sleep       */
#define EL_HEAD_F   0xF0   /* circadian   */
#define EL_HEAD_SYS 0x00

/* Flags bitmask */
#define EL_FLAG_SUSTAINED   (1 << 0)   /* sustained event (>= min duration) */
#define EL_FLAG_CONFIRMED   (1 << 1)   /* confirmed by second head */
#define EL_FLAG_CALIBRATING (1 << 2)   /* during baseline calibration period */

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t sequence;
    int64_t  timestamp_s;
    uint8_t  event_type;
    uint8_t  severity;
    uint16_t flags;
    float    features[10];
    float    anomaly_score;
    float    aux_float;
    uint8_t  head_id;
    uint8_t  reserved[59];
} ELRecord;

/* Compile-time size check */
static_assert(sizeof(ELRecord) == EL_RECORD_SIZE,
              "ELRecord must be exactly 128 bytes");

typedef struct {
    uint32_t total_records;
    uint32_t bytes_used;
    uint32_t bytes_free;
    uint32_t next_sequence;
} ELStatus;

/**
 * Mount the eventlog SPIFFS partition and initialise the logger.
 * Creates the log file if it does not exist.
 * Returns 0 on success.
 */
int el_init(void);

/**
 * Append one event record to the log.
 * timestamp_s: unix epoch (from SNTP or RTC)
 * Returns 0 on success, -1 on partition full or I/O error.
 */
int el_append(ELEventType type, uint8_t severity, uint16_t flags,
              uint8_t head_id, const float* features,
              float anomaly_score, float aux_float,
              int64_t timestamp_s);

/**
 * Read a record by index (0 = oldest).
 * Returns 0 on success, -1 if index out of range.
 */
int el_read(uint32_t index, ELRecord* out);

/**
 * Get log status (record count, space used/free).
 */
ELStatus el_status(void);

/**
 * Iterate all records from oldest to newest.
 * Calls cb(record, user_data) for each. Stops if cb returns false.
 */
void el_iterate(bool (*cb)(const ELRecord*, void*), void* user_data);

/**
 * Erase the entire log (factory reset only).
 */
int el_erase(void);

#ifdef __cplusplus
}
#endif