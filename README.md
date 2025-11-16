# Signify: An Embedded Assistive Traffic Sign Recognition System

**An assistive system for drivers, leveraging on-sensor AI inference to provide real-time audio alerts for traffic signs.**

This is a 4th-Year Computer Science project demonstrating a low-power, high-performance edge computing pipeline for a real-world Advanced Driver-Assistance System (ADAS).

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-blueviolet)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C30039?logo=raspberrypi)
![Sony IMX500](https://img.shields.io/badge/Sony-IMX500-lightgrey)

---

## 1. Problem Statement

In fast-paced or low-visibility driving conditions, drivers can momentarily miss critical traffic signs. This risk is compounded for individuals with visual impairments. This project aims to solve this problem by creating a low-cost, low-power, and highly responsive assistive system to mitigate this risk by detecting signs and announcing them via audio.

## 2. Key Features

* **On-Sensor Inference:** The core technical achievement. The quantized YOLOv8n model runs directly on the Sony IMX500's internal AI accelerator, not the Raspberry Pi's CPU.
* **Real-Time Performance:** The edge architecture frees the Pi's CPU, allowing the system to process detections and stream video with minimal latency.
* **High-Accuracy Model:** Trained on a diverse, region-specific dataset of 76 traffic sign classes (19,000+ images), achieving a final mAP50-95 of 69.5%.
* **Audio Alerts (TTS):** Uses Google's Text-to-Speech (`gTTS`) engine for natural, high-quality voice announcements (e.g., "Stop", "Speed Limit 50").
* **Intelligent Logic:**
    * **Label Cleaning:** Cleans complex labels (e.g., `regulatory--no_stopping` -> "no stopping").
    * **Smart Cooldown:** Prevents "spam" by only announcing a specific sign type only once every few seconds, even if it remains in view.
    * **Bluetooth Wakeup Fix:** Programmatically plays a 0.5-second silent MP3 *before* each announcement to "wake up" sleeping Bluetooth speakers, preventing the first syllable (e.g., the "s" in "stop") from being cut off.
* **Live Video Stream:** Hosts a Flask web server that streams the live, annotated video feed (with bounding boxes) to any browser on the network for monitoring or demonstration.

## 3. System Architecture & Diagrams

The system is a two-part architecture:

1.  **AI Camera (Sony IMX500):** This is the edge processor. It captures the image, runs the entire YOLOv8n model, and generates the detection results (tensors).
2.  **Host (Raspberry Pi 5):** This is the system orchestrator. It receives the tensors, post-processes them (NMS, filtering), manages the application logic (TTS state, cooldowns), and hosts the web stream.

---

<details>
<summary><b>View Deployment Diagram (Graphviz DOT Code)</b></summary>

*This diagram shows the physical and logical deployment of all hardware and software components. (To view, render the DOT code below or see the [image](docs/images/deployment_diagram.png)).*

<!-- Uncomment this line after adding the screenshot: -->
<!-- ![Deployment Diagram](docs/images/deployment_diagram.png) -->

```dot
digraph DeploymentDiagram {
    // --- Global Settings ---
    graph [
        label="Traffic Sign Detection System - Deployment View", labelloc=t, fontsize=24, fontname="Arial",
        rankdir=TB, splines=ortho, nodesep=0.8, ranksep=1.2, bgcolor="transparent", labelfontsize=14
    ];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=12];
    edge [fontname="Arial", fontsize=10, color="#333333", arrowhead=open, labelfontsize=11];

    // --- Node Definitions (Hardware as Clusters) ---
    subgraph cluster_RPi5 {
        label = "Raspberry Pi 5"; bgcolor = "lightblue"; style = "filled,rounded"; node [fillcolor=lightblue];
        RPi5_App [label="Traffic Sign App\n(Python/Flask/gTTS)", shape=component, fillcolor=white, style="filled,rounded"];
        RPi5_Mpg123 [label="Audio Player\n(mpg123)", shape=component, fillcolor=white, style="filled,rounded"];
        RPi5_PyEnv [label="Python Env.", shape=folder, fillcolor=honeydew, style="filled"];
        RPi5_BTMod [label="Bluetooth", shape=hexagon, style=filled, fillcolor=lightgrey, width=0.6];
        RPi5_CSI [label="MIPI CSI", shape=hexagon, style=filled, fillcolor=lightgrey, width=0.6];
        RPi5_Eth0 [label="Ethernet", shape=hexagon, style=filled, fillcolor=lightgrey, width=0.6];
    }
    subgraph cluster_AICam {
        label = "AI Camera (IMX500)"; bgcolor = "lightgreen"; style = "filled,rounded"; node [fillcolor=lightgreen];
        AICam_Model [label="Quantized Model\n(YOLOv8n)", shape=component, fillcolor=white, style="filled,rounded"];
        AICam_AIAccel [label="AI Accelerator", shape=cylinder, fillcolor=honeydew, style="filled"];
        AICam_Sensor [label="Sensor/ISP", shape=cylinder, fillcolor=honeydew, style="filled"];
        AICam_CamCSI [label="MIPI CSI", shape=hexagon, style=filled, fillcolor=lightgray, width=0.6];
    }
    subgraph cluster_External {
        label = "External Interfaces & Services"; style=invis;
        subgraph cluster_Laptop {
            label = "Laptop (Viewer)"; bgcolor = "lightgrey"; style = "filled,rounded";
            Laptop_Browser [label="Web Browser", shape=component, fillcolor=white, style="filled,rounded"];
            Laptop_Eth [label="Ethernet", shape=hexagon, style=filled, fillcolor=white, width=0.6];
        }
        subgraph cluster_Speaker {
            label = "Bluetooth Speaker"; bgcolor = "lightgrey"; style = "filled,rounded";
            Speaker_Audio [label="Audio Output", shape=component, fillcolor=white, style="filled,rounded"];
        }
        subgraph cluster_GCloud {
            label="Google Cloud"; bgcolor="lightyellow"; style = "filled,rounded";
            GCloud_GTTS [label="gTTS Service", shape=tab, fillcolor=white, style="filled"];
        }
    }

    // --- Layout Constraints ---
    { rank=same; cluster_RPi5; cluster_AICam; }
    { rank=same; cluster_Laptop; cluster_Speaker; cluster_GCloud; }

    // --- Connections (Using xlabel for better label positioning) ---
    RPi5_CSI -> AICam_CamCSI [ dir=both, xlabel=" <<MIPI CSI>> Physical Link ", color="darkgreen", style=bold, penwidth=2.0, constraint=false, arrowhead=none, arrowtail=none ];
    AICam_Sensor -> AICam_AIAccel [ xlabel=" Image Tensor ", style=solid, penwidth=1.5, color="black", arrowhead=normal ];
    AICam_AIAccel -> RPi5_App [ xlabel=" Detection Results (Tensors) ", style=dashed, color="purple", arrowhead=normal ];
    RPi5_App -> RPi5_PyEnv [ xlabel=" <<executes on>> ", style=dotted, arrowhead=open, color="dimgray" ];
    RPi5_App -> RPi5_Mpg123 [ xlabel=" <<uses>> ", style=dotted, arrowhead=open, color="dimgray" ];
    RPi5_Eth0 -> Laptop_Eth [ dir=both, xlabel=" <<Ethernet>> HTTP Stream ", color="darkorange", style=bold, penwidth=2.0, arrowhead=none, arrowtail=none ];
    RPi5_BTMod -> Speaker_Audio [ xlabel=" <<Bluetooth>> Audio Stream ", style=dashed, color="blue", penwidth=1.5, arrowhead=normal ];
    RPi5_App -> GCloud_GTTS [ xlabel=" 1. HTTPS Text Request ", style=dashed, color="red", arrowhead=vee ];
    GCloud_GTTS -> RPi5_App [ xlabel=" 2. HTTPS MP3 Response ", style=dashed, color="red", arrowhead=vee ];
    AICam_Model -> AICam_AIAccel [ xlabel=" <<deploy>> ", style=dotted, arrowhead=open, color="darkorange", penwidth=1.5 ];
}

</details>

<details> <summary><b>View Sequence Diagram (Mermaid Code)</b></summary>

This diagram shows the time-based interaction between components. (To view, render the Mermaid code below or see the image).

<!-- Uncomment this line after adding the screenshot: -->

<!-- -->
Code snippet

sequenceDiagram
    actor Driver
    box Traffic Sign System #LightBlue
        participant RPi5 as Raspberry Pi 5
        participant AICam as AI Camera (IMX500)
        participant AIAccel as AI Accelerator (on Cam)
        participant TTS as Text-to-Speech (gTTS/mpg123)
        participant BTSpkr as Bluetooth Speaker
    end

    Note over Driver, BTSpkr: Vehicle Moves, Sign Enters View
    Driver->>+AICam: Provides Visual Input (Implicit)
    
    Note over AICam, RPi5: --- Detailed Processing Begins ---
    activate AICam
        Note left of AICam: Capture Image
        AICam->>+AIAccel: Input Tensor (via ISP)
        activate AIAccel
            Note over AIAccel: On-Sensor Inference (using loaded weights)
            AIAccel-->>-AICam: Output Tensors (Raw Detections)
        deactivate AIAccel
        AICam-->>-RPi5: Sends Output Tensors
    deactivate AICam

    activate RPi5
        Note over RPi5: 1. Decode Tensors & Post-process (NMS)
        RPi5->>RPi5: 2. Filter by Confidence Threshold
        RPi5->>RPi5: 3. Check TTS Logic (Cooldown, History)

        opt Sign should be announced
            RPi5->>+TTS: Cleaned Label (e.g., "Stop")
            Note over RPi5, TTS: (Runs TTS in separate thread)
            activate TTS
                 Note over TTS: Generate MP3 (gTTS requires Internet)
                 TTS->>+BTSpkr: Play MP3 (via mpg123)
                 BTSpkr-->>-Driver: ("Stop")
             deactivate TTS
             RPi5->>RPi5: Update Announcement History
        end

        Note over RPi5: 4. Annotate Video Frame for Stream
        RPi5-->>Driver: Send Annotated Frame (MJPEG)
    deactivate RPi5
    Note over AICam, RPi5: --- Detailed Processing Ends ---

</details>

<details> <summary><b>View Activity Diagram (Mermaid Code)</b></summary>

This diagram shows the control flow and logic on the Raspberry Pi 5. (To view, render the Mermaid code below or see the image).

<!-- Uncomment this line after adding the screenshot: -->

<!-- -->
Code snippet

graph TD
    A[Receive Tensors] --> B{Decode};
    B --> C{Filter};
    C --> D{Detections?};

    %% No Path
    D -- No --> E[Annotate Empty];
    E --> N[Encode];

    %% Yes Path
    D -- Yes --> F{Check Cooldown};
    F -- No --> G[Annotate Existing];
    F -- Yes --> H{Check History};
    H -- Yes --> G;
    H -- No --> I[Clean Label];
    I --> J[Trigger TTS];
    J --> K[Update History];
    K --> L[Annotate New];

    %% Connect Yes path ends to Encode
    G --> N;
    L --> N;

    %% Continue
    N --> O[Stream Frame];
    O --> A; % Loop back

</details>

4. Final Training Results

The model was trained on the "Traffic Signs Dataset (Mapillary and DFG)" for the Africa region, which contains 76 classes and over 19,000 images. The training ran for 150 epochs on an NVIDIA RTX 4070 Laptop GPU.

Final best.pt Model Performance (on validation set):

    mAP50-95: 0.695 (69.5%)

    mAP50: 0.769 (76.9%)

    Precision: 0.858 (85.8%)

    Recall: 0.676 (67.6%)

Training Metrics

Confusion Matrix

5. Setup and Installation

This repository contains the code for both the training environment (on a powerful laptop) and the deployment environment (on the Raspberry Pi).

A. Training Environment (Laptop)

This environment is used to train, validate, and quantize the model.

    Clone Repository:
    Bash

git clone [https://github.com/is-project-4th-year/signify-traffic-sign-recognition-system.git](https://github.com/is-project-4th-year/signify-traffic-sign-recognition-system.git)
cd signify-traffic-sign-recognition-system

Set up Python Environment: It is highly recommended to use a virtual environment like Conda or venv.
Bash

conda create -n traffic_signs python=3.9
conda activate traffic_signs

Install Requirements:
Bash

pip install -r requirements.txt

Download Dataset:

    Download the Traffic Signs Dataset (Mapillary and DFG) from Kaggle.

    Unzip and place its contents (images, labels, etc.) into the dataset/ directory (as defined in .gitignore, this folder is not tracked).

Run Data Splitting:

    The dataset does not come pre-split. Run the provided script to split the data into train/ and val/ sets.

Bash

python scripts/split_dataset.py

Run Training:

    This script will use your GPU (if device=0 is set) to train the model.

Bash

bash scripts/train_model.sh

Quantize Model:

    After training, this script takes the best.pt file and exports it to the packerOut.zip format required by the IMX500.

Bash

    python scripts/export_imx500_africa.py

B. Deployment Environment (Raspberry Pi 5)

This is the runtime environment that is mounted in the vehicle.

    Install Dependencies: First, ensure your Pi has an internet connection.
    Bash

sudo apt update
sudo apt install espeak libespeak1 ffmpeg mpg123 -y

# Install Python packages
pip install gTTS flask opencv-python pydub --break-system-packages

# Install Sony's library
pip install git+[https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git)

Transfer Files to Pi:

    From your laptop, scp the necessary files to your Pi's home directory:

    app.py (This is the final script, e.g., app8_pydub_fix.py renamed to app.py)

    The quantized model: packerOut.zip (from your laptop's runs/.../weights/best_imx_model/ folder)

    The labels file: labels.txt (from the same folder as packerOut.zip)

Bash

scp path/to/app.py dgara@<PI_IP_ADDRESS>:~/
scp path/to/packerOut.zip dgara@<PI_IP_ADDRESS>:~/finalyearproject2/
scp path/to/labels.txt dgara@<PI_IP_ADDRESS>:~/finalyearproject2/

(Note: Ensure the paths inside app.py point to the correct locations, e.g., /home/dgara/finalyearproject2/packerOut.zip)

Run the Application:

    Connect your Bluetooth speaker to the Raspberry Pi.

    Run the main application script:

Bash

    python app.py

    View the Stream:

        On your laptop (or any device on the same network), open a web browser and go to: http://<PI_IP_ADDRESS>:8000 (e.g., http://10.42.0.2:8000)

6. License

This project is licensed under the MIT License. See the LICENSE file for full details.

