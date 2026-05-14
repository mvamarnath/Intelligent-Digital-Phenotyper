# Intelligent Diagnostic Phenotyper (IDP)

> **Edge-AI wearable on ESP32-S3 that continuously monitors physiology using five offline ML heads to detect autonomic anomalies, tremor, sleep disruption, and circadian shifts — logging structured clinical evidence without cloud dependency or diagnostic claims.**

---

## ⚠️ Important Disclaimer

This device is **not a medical diagnostic tool**. It does not diagnose, treat, or screen for any medical condition. All output is a structured physiological log intended to support — not replace — clinical evaluation. All interpretation must be performed by a licensed clinician.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Hardware](#hardware)
- [Software Stack](#software-stack)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Python Environment (Training)](#python-environment-training)
  - [ESP32-S3 Firmware](#esp32-s3-firmware)
- [Flashing Models to Device](#flashing-models-to-device)
- [Documentation](#documentation)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

The IDP is a fully offline, battery-operated wrist-worn device that continuously records physiological and behavioural patterns associated with neurological and psychiatric conditions. Unlike cloud-dependent wearables, every inference step runs locally on the ESP32-S3 microcontroller — no Wi-Fi, no phone tether, no external compute.

**What it does:**
- Extracts a 10-element feature vector every 60 seconds from PPG, accelerometer, and EDA sensors
- Runs five independent specialist ML heads for anomaly detection, stress classification, tremor detection, sleep staging, and circadian phase tracking
- Maintains a rolling 30-day personal baseline and flags deviations via Z-score
- Logs timestamped physiological events to an encrypted append-only SPIFFS partition
- Exports a structured clinical report via BLE when the user visits a physician

**What it does NOT do:**
- Diagnose any condition
- Transmit data to the cloud
- Output DSM-5 / ICD-11 diagnostic labels
- Replace clinical assessment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ESP32-S3  (offline)                      │
│                                                              │
│  Sensors ──► DSP Layer ──► Feature Vector (10D, 60s)        │
│  MAX30102        IIR filters      ┌──────────────────────┐  │
│  MPU9250         FFT (ESP-DSP)    │  Head B  Tremor       │  │
│  CJMCU-6701      ENMO, RMSSD      │  Head C  Autoencoder  │  │
│                  SCL, SCR         │  Head D  Stress clf   │  │
│                                   │  Head E  Sleep        │  │
│                                   │  Head F  Circadian    │  │
│                                   └──────────────────────┘  │
│                        │                                     │
│               Personal Baseline ──► Z-score ──► Event Log   │
│               (Welford online)       SPIFFS    (128B/record) │
│                                                              │
│                    BLE Export ──► Clinician Report           │
└─────────────────────────────────────────────────────────────┘
```

### Specialist Heads

| Head | Method | Dataset | Output | Size |
|------|--------|---------|--------|------|
| B — Tremor | Spectral rule engine (deterministic) | Synthetic validation | Class 0–3 (no/PD/ET/indeterminate) | 0 KB |
| C — Autoencoder | TFLM INT8 MLP (254 params) | WESAD baseline | Anomaly score (AUROC 0.874) | 3.0 KB |
| D — Stress clf | TFLM INT8 MLP (339 params) | WESAD all labels | 3-class probability (66.4% acc) | 2.9 KB |
| E — Sleep | van Hees HDCZA (deterministic) | Published params | Sleep/wake per 30 s epoch | 0 KB |
| F — Circadian | Forger 1999 ODE (deterministic) | Published params | DLMO + phase Z-score | 0 KB |

---

## Hardware

| Component | Part | Interface | Purpose |
|-----------|------|-----------|---------|
| Microcontroller | ESP32-S3 DevKitC-1 | — | 16 MB flash, 8 MB PSRAM |
| PPG sensor | MAX30102 | I2C (0x57) | BVP / heart rate |
| IMU | MPU9250 | I2C (0x68) | 3-axis accelerometer at 32 Hz |
| GSR sensor | CJMCU-6701 | ADC (GPIO1) | Electrodermal activity |

**Wiring (I2C shared bus):**
```
ESP32-S3 GPIO8  ──── SDA ──── MAX30102 + MPU9250
ESP32-S3 GPIO9  ──── SCL ──── MAX30102 + MPU9250
ESP32-S3 GPIO1  ──── OUT ──── CJMCU-6701
3.3V  ──────────────────────── VCC (all sensors)
GND   ──────────────────────── GND (all sensors)

I2C pull-ups: 4.7 kΩ from SDA → 3.3V and SCL → 3.3V (required)
```

---

## Software Stack

### Training (Python)
- Python 3.11
- TensorFlow 2.15 (model training + TFLite export)
- NumPy 1.26, Pandas, SciPy, scikit-learn, Matplotlib

### Firmware (C++)
- ESP-IDF v5.3.1
- C++17 (gnu++23 dialect)
- [espressif/esp-tflite-micro](https://github.com/espressif/esp-tflite-micro) v1.3.5
- [espressif/esp-dsp](https://github.com/espressif/esp-dsp) v1.8.2
- NimBLE (bundled with ESP-IDF)

---

## Datasets

Datasets are **not included** in this repository. Download them separately and place them in `wearable_project/` as shown in [Project Structure](#project-structure).

| Dataset | Used for | Access | Citation |
|---------|----------|--------|----------|
| **WESAD** | Head C + D training (autonomic autoencoder, stress classifier) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) / [Direct mirror](https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download) | Schmidt et al., ICMI 2018 |
| **DEPRESJON** | Activity features, sleep pattern reference | [Simula Datasets](https://datasets.simula.no/depresjon/) / [Kaggle](https://www.kaggle.com/datasets/arashnic/the-depression-dataset) | Garcia-Ceja et al., ACM MMSys 2018 |
| **PSYKOSE** | Activity features, schizophrenia motor patterns | [Simula Datasets](https://datasets.simula.no/psykose/) / [OSF](https://osf.io/dgjzu/) | Jakobsen et al., IEEE CBMS 2020 |
| **Levodopa Response Study** | Head B tremor classifier (planned) | [Synapse syn20681023](https://www.synapse.org/#!Synapse:syn20681023) (free account + DUA) | Vergara-Diaz et al., Sci Data 2021 |
| **EmpaticaE4Stress** | Cross-dataset stress classifier validation | [Mendeley Data](https://data.mendeley.com/datasets/g2p7vwxyn2/1) | Campanella et al., Data in Brief 2024 |

---

## Project Structure

```
wearable_project/
├── datasets/
│   ├── WESAD/                  # 15 subject folders S2–S17 (.pkl files)
│   ├── depresjon/data/         # condition/ + control/ CSVs + scores.csv
│   ├── psykose/                # patient/ + control/ CSVs + patients_info.csv
│   └── (other datasets)
│
├── code/                       # Python training pipeline
│   ├── config.py               # Central path configuration
│   ├── data_loaders.py         # WESADLoader, DepresjonLoader, PsykoseLoader
│   ├── features.py             # 10-element feature extractor (v1)
│   ├── extract_wesad_all.py    # Batch WESAD feature extraction
│   ├── extract_activity_all.py # DEPRESJON + PSYKOSE extraction
│   ├── train_autoencoder.py    # Head C: LOSO training + TFLite export
│   ├── train_stress_classifier.py  # Head D: LOSO training + TFLite export
│   ├── sleep_classifier.py     # Head E: van Hees validation
│   ├── circadian_tracker.py    # Head F: Forger ODE validation
│   └── tremor_classifier.py    # Head B: spectral rule engine + tests
│
├── features/                   # Extracted feature matrices (.npz)
│   ├── wesad_features.npz      # X(741,10), y(741,), subject, feature_names
│   └── activity_features.npz  # X(52056,8), y(52056,), dataset
│
├── models/                     # Trained model files
│   ├── autoencoder_int8.tflite         # Head C (3.0 KB)
│   ├── stress_clf_int8.tflite          # Head D (2.9 KB)
│   ├── autoencoder_quant_params.h      # C header: preprocessing constants
│   ├── stress_clf_quant_params.h       # C header: preprocessing constants
│   └── tremor_params.h                 # C header: spectral thresholds
│
├── reports/                    # Verification plots (.png)
│
├── firmware/                   # ESP32-S3 firmware (ESP-IDF project)
│   ├── CMakeLists.txt
│   ├── partitions.csv          # 16 MB flash layout
│   ├── sdkconfig.defaults      # PSRAM, BLE, SPIFFS config
│   ├── main/
│   │   ├── main.cpp            # app_main, sensor_task, analysis_task
│   │   ├── dsp/                # IIR filters, FFT wrapper, feature extractor
│   │   ├── heads/              # Head B–F implementations
│   │   ├── storage/            # Event logger, baseline store
│   │   ├── ble/                # NimBLE GATT export server
│   │   └── sensors/            # MAX30102, MPU9250, CJMCU-6701 drivers
│   └── managed_components/     # esp-tflite-micro, esp-dsp (auto-fetched)
│
└── docs/
    ├── idp_documentation_1.pdf  # Phase 1: Data pipeline & feature extraction
    ├── idp_documentation_2.pdf  # Phase 2: Model training & TFLite export
    ├── idp_documentation_3.pdf  # Phase 3: ESP32-S3 firmware architecture
    └── idp_documentation_4.pdf  # Phase 4: Sensor drivers & hardware integration
```

---

## Getting Started

### Python Environment (Training)

> Requires Python 3.11. Python 3.12/3.13 not supported (TensorFlow 2.15 Windows wheels).

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/idp-wearable.git
cd idp-wearable

# Create virtual environment
py -3.11 -m venv code/.venv
code/.venv/Scripts/Activate.ps1      # Windows
# source code/.venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r code/requirements.txt

# Install TensorFlow (CPU-only)
pip install tensorflow==2.15.0

# Verify datasets are in place
python code/verify_data.py
```

**requirements.txt:**
```
numpy>=1.26,<2.0
pandas>=2.1
scipy>=1.11
matplotlib>=3.8
tqdm>=4.66
scikit-learn>=1.3
```

**Training pipeline (run in order):**
```bash
python code/extract_wesad_all.py        # Extract WESAD features → wesad_features.npz
python code/extract_activity_all.py     # Extract activity features → activity_features.npz
python code/train_autoencoder.py        # Train + export Head C → autoencoder_int8.tflite
python code/train_stress_classifier.py  # Train + export Head D → stress_clf_int8.tflite
python code/tremor_classifier.py        # Validate Head B (synthetic)
python code/sleep_classifier.py         # Validate Head E on DEPRESJON/PSYKOSE
python code/circadian_tracker.py        # Validate Head F on DEPRESJON
```

---

### ESP32-S3 Firmware

**Prerequisites:**
- [ESP-IDF v5.3.1](https://docs.espressif.com/projects/esp-idf/en/v5.3.1/esp32s3/get-started/index.html) installed and on PATH
- ESP32-S3 DevKitC-1 (16 MB flash, 8 MB PSRAM)

```bash
cd firmware

# Fetch managed components (first time only)
idf.py add-dependency "espressif/esp-tflite-micro"
idf.py add-dependency "espressif/esp-dsp"

# Build
idf.py build

# Flash firmware
idf.py -p COM_PORT flash

# Monitor serial output
idf.py -p COM_PORT monitor
```

**Expected binary size:** ~708 KB / 3 MB app partition

---

## Flashing Models to Device

The `.tflite` model files must be flashed to a separate SPIFFS partition:

```bash
# Create models image
mkdir spiffs_models
copy models\autoencoder_int8.tflite spiffs_models\
copy models\stress_clf_int8.tflite  spiffs_models\

# Generate SPIFFS binary (1 MB partition)
python %IDF_PATH%\components\spiffs\spiffsgen.py 1048576 spiffs_models build\spiffs_models.bin

# Flash to models partition (offset 0x310000)
python -m esptool --chip esp32s3 -p COM_PORT -b 460800 write_flash 0x310000 build\spiffs_models.bin
```

**Expected serial output on successful boot:**
```
I (xxx) MAX30102: MAX30102 ready (PART_ID=0x15, HR mode, IR only)
I (xxx) MPU9250:  MPU9250 ready (WHO_AM_I=0x71, accel-only ±2g ~32 Hz)
I (xxx) GSR_EDA:  GSR/EDA ADC ready (GPIO1, ADC1_CH0, scale=1.00)
I (xxx) HeadC:    Model loaded: 3096 bytes — Ready. Threshold=2.0000
I (xxx) HeadD:    Model loaded: 2944 bytes — Ready
I (xxx) MAIN:     All subsystems initialised — Tasks running
```

---

## Documentation

Full technical documentation is in `docs/`. Each volume covers one phase of development:

| Volume | Phase | Contents |
|--------|-------|----------|
| [Volume 1](docs/idp_documentation_1.pdf) | Data Pipeline & Feature Engineering | Dataset selection, data loaders, 10-element feature extractor design, 4-iteration BVP peak detector development, Cohen's d validation |
| [Volume 2](docs/idp_documentation_2.pdf) | Model Training & TFLite Export | All 5 head implementations, LOSO validation results, INT8 quantisation, known limitations register |
| [Volume 3](docs/idp_documentation_3.pdf) | ESP32-S3 Firmware | Project scaffold, DSP layer, head C++implementations, event logger, BLE export, task architecture, 14-error build log |
| [Volume 4](docs/idp_documentation_4.pdf) | Sensor Drivers & Hardware Integration | MAX30102/MPU9250/CJMCU-6701 drivers, I2C bus manager, wiring guide, first-flash procedure, verification checklist |

---

## Results

### Model validation (WESAD dataset, LOSO cross-validation)

| Metric | Value |
|--------|-------|
| Head C — Anomaly detection AUROC (baseline vs stress) | 0.874 ± 0.142 |
| Head C — Best fold AUROC | 1.000 (S16) |
| Head D — Overall 3-class accuracy | 66.4% |
| Head D — Stress class recall | 83.6% |
| Head B — Synthetic test pass rate | 8/8 |

### Resource usage (ESP32-S3)

| Resource | Usage |
|----------|-------|
| App partition | 708 KB / 3 MB (78% free) |
| Head C model | 3.0 KB flash |
| Head D model | 2.9 KB flash |
| Tensor arena (per head) | 8 KB IRAM |
| Sensor circular buffers | ~43 KB PSRAM |
| Total PSRAM used | < 1 MB / 8 MB |

### Feature separability (Cohen's d, baseline vs stress, n=741 windows)

| Feature | Effect size | Direction |
|---------|-------------|-----------|
| SCR_COUNT | 1.03 (large) | stress higher ✓ |
| SDNN | 1.38 (large) | inverted (wrist PPG limitation) |
| SCL | 0.72 (medium) | stress higher ✓ |
| HR_MEAN | 0.79 (medium) | stress higher ✓ |
| TEMP_MEAN | 0.54 (medium) | stress lower ✓ (vasoconstriction) |

---

## Known Limitations

| ID | Limitation | Mitigation |
|----|-----------|------------|
| L-01 | Wrist PPG RMSSD unreliable under motion | Motion gate (NaN on high-ENMO windows) |
| L-02 | WESAD is lab data (15 healthy subjects) | Cross-validate on EmpaticaE4Stress |
| L-07 | Circadian phase inflated on activity-count data | ACC-mode HDCZA resolves on real hardware |
| L-09 | Sleep duration inflated by activity-count proxy | ACC-mode van Hees used on device |
| L-11 | Tremor classifier validated on synthetic signals only | Levodopa Response Study validation planned |

Full limitations register in [Volume 2](docs/idp_documentation_2.pdf) and [Volume 3](docs/idp_documentation_3.pdf).

---

## Roadmap

- [x] Phase 1 — Data pipeline and feature extraction
- [x] Phase 2 — Model training and TFLite export
- [x] Phase 3 — ESP32-S3 firmware architecture
- [x] Phase 4 — Sensor drivers (MAX30102, MPU9250, CJMCU-6701)
- [ ] Phase 5 — Hardware bring-up, calibration, self-experiment validation
- [ ] Cross-dataset validation on EmpaticaE4Stress
- [ ] Tremor classifier validation on Levodopa Response Study
- [ ] BLE companion app for clinical report export
- [ ] Power profiling and battery life optimisation
- [ ] Journal submission

---

## References

1. Schmidt P, et al. *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018. [doi:10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)
2. Elgendi M, et al. *Systolic Peak Detection in Acceleration Photoplethysmograms.* PLOS ONE 2013. [doi:10.1371/journal.pone.0076585](https://doi.org/10.1371/journal.pone.0076585)
3. van Hees VT, et al. *Estimating sleep parameters using an accelerometer without sleep diary.* Sci Rep 2018. [doi:10.1038/s41598-018-31266-z](https://doi.org/10.1038/s41598-018-31266-z)
4. Forger DB, Jewett ME, Kronauer RE. *A simpler model of the human circadian pacemaker.* J Biol Rhythms 1999;14(6):532–537.
5. Lim D, et al. *Accurately predicting mood episodes using wearable sleep and circadian rhythm features.* npj Digital Medicine 2024. [doi:10.1038/s41746-024-01333-z](https://doi.org/10.1038/s41746-024-01333-z)
6. Garcia-Ceja E, et al. *Depresjon: A Motor Activity Database of Depression Episodes.* ACM MMSys 2018. [doi:10.1145/3204949.3208125](https://doi.org/10.1145/3204949.3208125)
7. Jakobsen P, et al. *PSYKOSE: A Motor Activity Database of Patients with Schizophrenia.* IEEE CBMS 2020. [doi:10.1109/CBMS49503.2020.00064](https://doi.org/10.1109/CBMS49503.2020.00064)
8. Vergara-Diaz G, et al. *Accelerometer data from wearable sensors for Parkinson's disease.* Sci Data 2021. [doi:10.1038/s41597-021-00830-0](https://doi.org/10.1038/s41597-021-00830-0)

---

## License

This project is released under the [MIT License](LICENSE).

The datasets used for training are subject to their own licenses:
- WESAD: Academic / non-commercial use
- DEPRESJON: CC BY 4.0
- PSYKOSE: CC BY 4.0
- EmpaticaE4Stress: CC BY 4.0
- Levodopa Response Study: Synapse Data Use Agreement (free, academic)

---

*Built with ESP-IDF, TensorFlow Lite Micro, and open wearable datasets. No cloud. No diagnosis. Just data.*
