#!/bin/bash
# This script runs the YOLOv8 training using the GPU.

echo "--- Starting YOLOv8 Training on GPU ---"

# Change directory to the main project folder first
# This ensures relative paths in the YAML and project output work correctly
cd "$(dirname "$0")/.." || exit 1 # Go up one level from 'scripts'

yolo train \
    model=yolov8n.pt \
    data=dataset/dataset.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    project=runs \
    name=africa_model_gpu_v1 \
    patience=50 \
    cache=True \
    workers=8 \
    device=0  

echo "Training Complete"
echo "Results saved in: runs/africa_model_gpu_v1/"
