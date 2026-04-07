#!/bin/bash
# Training with optimized parameters for better convergence

.venv\Scripts\yolo.exe train \
    data=dataset_v2/dataset.yaml \
    model=yolov8n.pt \
    epochs=80 \
    imgsz=640 \
    batch=16 \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    warmup_momentum=0.8 \
    warmup_bias_lr=0.1 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=0.0 \
    translate=0.1 \
    scale=0.5 \
    shear=0.0 \
    perspective=0.0 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.1 \
    copy_paste=0.0 \
    workers=8 \
    project=runs/detect \
    name=joystick_v2_optimized