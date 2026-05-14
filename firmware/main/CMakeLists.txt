idf_component_register(
    SRCS
        "main.cpp"
        "dsp/feature_extractor.cpp"
        "dsp/iir_filter.cpp"
        "dsp/fft_wrapper.cpp"
        "heads/head_b_tremor.cpp"
        "heads/head_c_autoencoder.cpp"
        "heads/head_d_stress.cpp"
        "heads/head_e_sleep.cpp"
        "heads/head_f_circadian.cpp"
        "storage/event_logger.cpp"
        "storage/baseline_store.cpp"
        "ble/ble_export.cpp"
        "sensors/i2c_bus.cpp"
        "sensors/max30102.cpp"
        "sensors/mpu9250.cpp"
        "sensors/gsr_eda.cpp"

    INCLUDE_DIRS
        "."
        "dsp"
        "heads"
        "storage"
        "ble"
        "sensors"

    REQUIRES
        esp_timer
        spiffs
        nvs_flash
        bt
        driver
        esp-dsp
        esp_driver_i2c
        esp_adc
)

# TFLM needs these flags
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O2
    -fno-exceptions
    -fno-rtti
    -DTF_LITE_STATIC_MEMORY
    -DESP_NN
)