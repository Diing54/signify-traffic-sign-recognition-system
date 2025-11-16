# Signify: An Embedded Assistive Traffic Sign Recognition System

**A 4th-Year Computer Science project demonstrating a real-time, on-sensor AI system for detecting and announcing traffic signs to assist drivers.**

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-blueviolet)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C30039?logo=raspberrypi)
![Sony IMX500](https://img.shields.io/badge/Sony-IMX500-lightgrey)

---

## 1. Project Overview

This project addresses the challenge of driver inattention or visual impairment by creating a low-cost, low-power assistive system. It uses a quantized YOLOv8n model to detect 76 unique, region-specific traffic signs in real-time and provides clear, spoken audio alerts to the driver.

## 2. Key Features

- **On-Sensor Inference:** The core technical achievement. The quantized YOLOv8n model runs directly on the Sony IMX500's internal AI accelerator, not the Raspberry Pi's CPU.
- **Real-Time Performance:** This edge architecture frees the Pi's CPU, allowing the system to process detections and stream video with minimal latency.
- **High-Accuracy Model:** Trained on a diverse, region-specific dataset of 76 traffic sign classes (19,000+ images), achieving a final **mAP50-95 of 69.5%**.
- **Audio Alerts (TTS):** Uses Google's Text-to-Speech (`gTTS`) for natural, high-quality voice announcements (e.g., "Stop", "Speed Limit 50").
- **Intelligent Logic:**
    - **Label Cleaning:** Cleans complex labels (e.g., `regulatory--no_stopping` -> "no stopping").
    - **Smart Cooldown:** Prevents "spam" by only announcing a specific sign type once every few seconds.
    - **Bluetooth Wakeup Fix:** Programmatically plays a 0.5-second silent MP3 *before* each announcement to "wake up" sleeping Bluetooth speakers, preventing the first syllable (e.g., the "s" in "stop") from being cut off.
- **Live Video Stream:** Hosts a Flask web server that streams the live, annotated video feed (with bounding boxes) to any browser on the network for monitoring or demonstration.

## 3. System Architecture

The system uses a two-part architecture:

1.  **AI Camera (Sony IMX500):** This is the edge processor. It captures the image, runs the entire YOLOv8n model using the deployed `packerOut.zip` file, and generates the detection results (tensors).
2.  **Host (Raspberry Pi 5):** This is the system orchestrator. It receives the tensors, post-processes them (NMS, filtering), manages the application logic (TTS state, cooldowns), and hosts the web stream.

---

## 4. Final Training Results

The model was trained on the **"Traffic Signs Dataset (Mapillary and DFG)"** for the Africa region (76 classes, 19k+ images) for 150 epochs.

**Final `best.pt` Model Performance (on validation set):**
* **mAP50-95 (Primary Metric):** `0.695 (69.5%)`
* **mAP50:** `0.769 (76.9%)`
* **Precision:** `0.858 (85.8%)`
* **Recall:** `0.676 (67.6%)`

### Training Metrics (from `runs/africa_model_gpu_v1`)
These graphs show the model's learning progress over the 150 epochs. The loss functions trend downwards (good) while the mAP/Precision/Recall metrics trend upwards (good).

![Training Metrics Graph](results/training_metrics.png)

### Confusion Matrix (from `runs/africa_model_gpu_v1`)
This matrix shows the model's performance on a per-class basis. Strong diagonal values indicate high accuracy for those classes.

![Confusion Matrix](results/confusion_matrix.png)

---

## 5. Setup and Installation

This repository contains the code for both the training environment (on a powerful laptop) and the deployment environment (on the Raspberry Pi).

### A. Training Environment (Laptop)

This environment is used to train, validate, and quantize the model.

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/is-project-4th-year/signify-traffic-sign-recognition-system.git](https://github.com/is-project-4th-year/signify-traffic-sign-recognition-system.git)
    cd signify-traffic-sign-recognition-system
    ```
2.  **Set up Python Environment:**
    *It is highly recommended to use a virtual environment like Conda or venv.*
    ```bash
    conda create -n traffic_signs python=3.9
    conda activate traffic_signs
    ```
3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Dataset:**
    * Download the [Traffic Signs Dataset (Mapillary and DFG)](https://www.kaggle.com/datasets/nomihsa965/traffic-signs-dataset-mapillary-and-dfg) from Kaggle.
    * Unzip and place its contents into a `dataset/` directory (this directory is ignored by Git via `.gitignore`).
5.  **Run Data Splitting:**
    * The dataset does not come pre-split. Run the provided script to split the data into `train/` and `val/` sets.
    ```bash
    python scripts/split_dataset.py
    ```
6.  **Run Training:**
    * This script will use your GPU (if `device=0` is set) to train the model.
    ```bash
    bash scripts/train_model.sh
    ```
7.  **Quantize Model:**
    * After training, this script takes the `best.pt` file and exports it to the `packerOut.zip` format required by the IMX500.
    ```bash
    python scripts/export_imx500_africa.py
    ```

### B. Deployment Environment (Raspberry Pi 5)

This is the runtime environment that is mounted in the vehicle.

1.  **Install Dependencies:**
    *First, ensure your Pi has an internet connection.*
    ```bash
    sudo apt update
    sudo apt install espeak libespeak1 ffmpeg mpg123 -y
    
    # Install Python packages
    pip install gTTS flask opencv-python pydub --break-system-packages
    
    # Install Sony's library
    pip install git+[https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git)
    ```
2.  **Transfer Files to Pi:**
    * From your laptop, `scp` the necessary files to your Pi's home directory:
    * `app.py` (This is the final script, e.g., `app8_pydub_fix.py` renamed to `app.py`)
    * The quantized model: `packerOut.zip` (from your laptop's `runs/.../weights/best_imx_model/` folder)
    * The labels file: `labels.txt` (from the same folder as `packerOut.zip`)
    ```bash
    scp path/to/app.py dgara@<PI_IP_ADDRESS>:~/
    scp path/to/packerOut.zip dgara@<PI_IP_ADDRESS>:~/finalyearproject2/
    scp path/to/labels.txt dgara@<PI_IP_ADDRESS>:~/finalyearproject2/
    ```
    *(Note: Ensure the paths inside `app.py` point to the correct locations, e.g., `/home/dgara/finalyearproject2/packerOut.zip`)*

3.  **Run the Application:**
    * Connect your Bluetooth speaker to the Raspberry Pi.
    * Run the main application script:
    ```bash
    python app.py
    ```
4.  **View the Stream:**
    * On your laptop (or any device on the same network), open a web browser and go to:
    **`http://<PI_IP_ADDRESS>:8000`**
    *(e.g., `http://10.42.0.2:8000`)*

---

## 6. License
This project is licensed under the **MIT License**. See the `LICENSE` file for full details.
