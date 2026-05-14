#include "event_logger.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "nvs_flash.h"
#include "nvs.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static const char* TAG = "EL";

/* ── State ───────────────────────────────────────────────────────────────── */
static bool     s_ready       = false;
static uint32_t s_next_seq    = 0;
static uint32_t s_n_records   = 0;

/* ── Placeholder cipher (XOR with device-unique key) ─────────────────────── 
 * Replace with mbedTLS AES-128-CTR in the security pass.               */
static const uint8_t kXorKey[16] = {
    0x49, 0x44, 0x50, 0x2D, 0x4B, 0x45, 0x59, 0x2D,
    0x56, 0x31, 0x2D, 0x50, 0x48, 0x2D, 0x30, 0x31
};

static void cipher_record(ELRecord* r)
{
    uint8_t* b = (uint8_t*)r;
    /* Skip magic and sequence (first 8 bytes) — needed for indexing */
    for (int i = 8; i < EL_RECORD_SIZE; i++) {
        b[i] ^= kXorKey[i % 16];
    }
}

/* ── NVS sequence counter ────────────────────────────────────────────────── */
static void load_seq(void)
{
    nvs_handle_t h;
    if (nvs_open(EL_NVS_NS, NVS_READONLY, &h) != ESP_OK) return;
    nvs_get_u32(h, EL_NVS_SEQ, &s_next_seq);
    nvs_close(h);
}

static void save_seq(void)
{
    nvs_handle_t h;
    if (nvs_open(EL_NVS_NS, NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_u32(h, EL_NVS_SEQ, s_next_seq);
    nvs_commit(h);
    nvs_close(h);
}

/* ── SPIFFS mount ────────────────────────────────────────────────────────── */
static int mount_spiffs(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path       = EL_MOUNT_POINT,
        .partition_label = "eventlog",
        .max_files       = 4,
        .format_if_mount_failed = true,
    };
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "SPIFFS mount failed: %d", ret);
        return -1;
    }
    return 0;
}

/* ── Record count from file size ─────────────────────────────────────────── */
static uint32_t count_records(void)
{
    struct stat st;
    if (stat(EL_FILE_PATH, &st) != 0) return 0;
    return (uint32_t)(st.st_size / EL_RECORD_SIZE);
}

/* ── Init ────────────────────────────────────────────────────────────────── */
int el_init(void)
{
    if (mount_spiffs() != 0) return -1;

    /* Create file if not present */
    FILE* f = fopen(EL_FILE_PATH, "ab");
    if (!f) { ESP_LOGE(TAG, "Cannot create log file"); return -1; }
    fclose(f);

    s_n_records = count_records();
    load_seq();

    /* Sync sequence counter with actual record count if needed */
    if (s_next_seq < s_n_records) s_next_seq = s_n_records;

    s_ready = true;
    ESP_LOGI(TAG, "Ready. Records: %lu  NextSeq: %lu",
             (unsigned long)s_n_records, (unsigned long)s_next_seq);
    return 0;
}

/* ── Append ──────────────────────────────────────────────────────────────── */
int el_append(ELEventType type, uint8_t severity, uint16_t flags,
              uint8_t head_id, const float* features,
              float anomaly_score, float aux_float,
              int64_t timestamp_s)
{
    if (!s_ready) return -1;

    ELRecord r = {};
    r.magic         = EL_MAGIC;
    r.sequence      = s_next_seq;
    r.timestamp_s   = timestamp_s;
    r.event_type    = (uint8_t)type;
    r.severity      = severity;
    r.flags         = flags;
    r.anomaly_score = anomaly_score;
    r.aux_float     = aux_float;
    r.head_id       = head_id;

    if (features) {
        for (int i = 0; i < 10; i++) r.features[i] = features[i];
    }

    cipher_record(&r);

    FILE* f = fopen(EL_FILE_PATH, "ab");
    if (!f) { ESP_LOGE(TAG, "Cannot open log for append"); return -1; }
    size_t written = fwrite(&r, 1, EL_RECORD_SIZE, f);
    fclose(f);

    if (written != EL_RECORD_SIZE) {
        ESP_LOGE(TAG, "Write failed: %u/%d bytes", (unsigned)written, EL_RECORD_SIZE);
        return -1;
    }

    s_next_seq++;
    s_n_records++;
    save_seq();

    ESP_LOGD(TAG, "Event logged: type=0x%02X seq=%lu",
             type, (unsigned long)(s_next_seq - 1));
    return 0;
}

/* ── Read ────────────────────────────────────────────────────────────────── */
int el_read(uint32_t index, ELRecord* out)
{
    if (!s_ready || index >= s_n_records) return -1;

    FILE* f = fopen(EL_FILE_PATH, "rb");
    if (!f) return -1;

    fseek(f, (long)(index * EL_RECORD_SIZE), SEEK_SET);
    size_t n = fread(out, 1, EL_RECORD_SIZE, f);
    fclose(f);

    if (n != EL_RECORD_SIZE) return -1;

    /* Decrypt */
    cipher_record(out);

    if (out->magic != EL_MAGIC) {
        ESP_LOGW(TAG, "Bad magic at index %lu", (unsigned long)index);
        return -1;
    }
    return 0;
}

/* ── Status ──────────────────────────────────────────────────────────────── */
ELStatus el_status(void)
{
    ELStatus st = {};
    st.total_records  = s_n_records;
    st.bytes_used     = s_n_records * EL_RECORD_SIZE;
    st.next_sequence  = s_next_seq;

    size_t total, used;
    if (esp_spiffs_info("eventlog", &total, &used) == ESP_OK) {
        st.bytes_free = (uint32_t)(total - used);
    }
    return st;
}

/* ── Iterate ─────────────────────────────────────────────────────────────── */
void el_iterate(bool (*cb)(const ELRecord*, void*), void* user_data)
{
    if (!s_ready || !cb) return;
    ELRecord r;
    for (uint32_t i = 0; i < s_n_records; i++) {
        if (el_read(i, &r) == 0) {
            if (!cb(&r, user_data)) break;
        }
    }
}

/* ── Erase ───────────────────────────────────────────────────────────────── */
int el_erase(void)
{
    FILE* f = fopen(EL_FILE_PATH, "wb");
    if (!f) return -1;
    fclose(f);
    s_n_records = 0;
    s_next_seq  = 0;
    save_seq();
    ESP_LOGW(TAG, "Log erased");
    return 0;
}