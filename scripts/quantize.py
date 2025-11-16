
import os
from pathlib import Path
from ultralytics import YOLO

# Path to the best.pt weights file from your GPU training run.
MODEL_PATH = Path("/home/astro/finalyearproject/runs/africa_model_gpu_v1/weights/best.pt")

# Path to your dataset's YAML file.
DATASET_YAML_PATH = Path("/home/astro/finalyearproject/dataset/dataset.yaml")

# Set the device for export (GPU recommended for speed).
DEVICE = '0' # Use '0' for GPU, 'cpu' for CPU

# Fraction of validation data for calibration (adjust if needed, start low).
CALIBRATION_FRACTION = 0.1 # 20%

def export_model_for_imx500():
    """
    Loads the trained Africa model and exports it to the Sony IMX500 format (INT8 PTQ).
    """
    # --- 2. VALIDATION ---
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        return
    if not DATASET_YAML_PATH.exists():
        print(f"❌ ERROR: Dataset YAML file not found at: {DATASET_YAML_PATH}")
        return

    print(f" Configuration seems correct. Starting the export process...")
    print(f"   - Model: {MODEL_PATH}")
    print(f"   - Dataset: {DATASET_YAML_PATH}")
    print(f"   - Using {CALIBRATION_FRACTION * 100}% of validation data for calibration.")

    # --- 3. MODEL LOADING ---
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f" ERROR: Failed to load the model. {e}")
        return

    # --- 4. EXPORT AND QUANTIZATION ---
    try:
        print("\n Starting export to IMX500 format...")
        print("   This may take a few minutes (quantization can be slow).")
        model.export(
            format='imx',
            data=DATASET_YAML_PATH,
            device=DEVICE,
            fraction=CALIBRATION_FRACTION
        )

        # --- 5. SUCCESS ---
        output_dir_name = f"{MODEL_PATH.stem}_imx_model" # e.g., best_imx_model
        # The output directory is relative to where the script is run OR relative to the model path depending on ultralytics version.
        # Let's assume it's created next to the best.pt file for clarity.
        output_path = MODEL_PATH.parent.parent / output_dir_name # Go up from weights/

        print("\n SUCCESS! Export complete.")
        print(f"   - Your exported model files are likely in: {output_path}")
        print(f"   - The key file for deployment is: {output_path}/packerOut.zip")

    except Exception as e:
        print(f" ERROR: An error occurred during the export process. {e}")


if __name__ == '__main__':
    export_model_for_imx500()
