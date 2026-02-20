# Object Detection on Raspberry Pi 4

This project provides two scripts for real-time object detection using SSD MobileNet v3 on a Raspberry Pi 4.

## Files

1.  **`object_detection_alert.py`**: Performs real-time object detection and displays a visual "ALERT!" on the screen when a person, dog, or cat is detected. Uses the CPU for processing.
2.  **`object_detection_cuda.py`**: A version optimized with CUDA acceleration and a lower input resolution for faster processing. Note that standard Raspberry Pi 4 hardware does **not** support CUDA; this script is intended for devices with NVIDIA GPUs or external accelerators.

## Prerequisites

-   Raspberry Pi 4 (or compatible)
-   Camera module (USB or Pi Camera)
-   Python 3 installed

## Installation

1.  **Update and Upgrade**:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2.  **Install OpenCV**:
    The scripts require OpenCV with DNN support.
    ```bash
    pip install opencv-python
    ```

3.  **Required Files**:
    Ensure the following files are in the same directory:
    - `coco.names` (Label mapping)
    - `frozen_inference_graph.pb` (Model weights)
    - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` (Model configuration)

## Running the Scripts

To run the standard detection with alerts:
```bash
python3 object_detection_alert.py
```

### Note on CUDA Script
If you try to run `object_detection_cuda.py` on a standard Raspberry Pi 4, it will likely fail with a `cv2.error` because the RPi 4 lacks an NVIDIA GPU. For Pi usage, **`object_detection_alert.py`** is the recommended script.

## Controls

-   Press **'q'** in the output window to exit the application.
