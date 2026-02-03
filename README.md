# Vision-Based Cooperative Perception for Intersection Tracking

This project is a prototype for a cooperative infrastructure perception system designed to track vehicles through a complex intersection using multiple camera views. It leverages the LUMPI dataset to perform object detection, project vehicle locations onto a unified top-down map, fuse data from different viewpoints, and perform real-time tracking.

***

## Core Components
* **Dataset:** [LUMPI (Multi-View Multi-Person Tracking in Urban Scenes)](https://data.uni-hannover.de/gl/dataset/lumpi)
* **Object Detection:** YOLOv8s (pre-trained)
* **Projection:** Homography Transformation (OpenCV)
* **Intra-Camera Fusion:** DBSCAN Clustering
* **Cross-Camera Fusion:** Centroid Averaging
* **Tracking:** Extended Kalman Filter (EKF)
* [cite_start]**Key Libraries:** PyTorch, Ultralytics, OpenCV, NumPy, Scikit-learn 

***

## Pipeline Overview

The system processes multiple camera streams in a sequential pipeline to produce a unified, tracked view of the intersection.

1.  **Multi-Camera Input**: The system ingests simultaneous, time-synchronized video streams from infrastructure-mounted cameras overlooking an intersection.

2.  **Object Detection**: For each frame from each camera, a pre-trained **YOLOv8s** model detects vehicles (`car`, `truck`) and generates axis-aligned 2D bounding boxes.

3.  **Point Extraction & De-duplication**: Instead of using the full box, a single representative point is extracted using a `bottom_center` heuristic. To handle redundant/overlapping detections of the same vehicle, **DBSCAN** is applied to cluster these points. The centroid of each cluster is then used as a single, stable detection for that camera view.

4.  **Homographic Projection**: A pre-computed **homography matrix** for each camera maps the extracted 2D points from the camera's perspective to the corresponding pixel coordinates on a top-down satellite image.

5.  **Cross-Camera Fusion**: The projected points from all cameras are fused. A distance-based clustering algorithm groups points that likely correspond to the same vehicle, and their positions are averaged to produce a single, fused detection.

6.  **State Estimation & Tracking**: The fused detections are fed as measurements into an **Extended Kalman Filter (EKF)**. The EKF maintains the state (position, velocity, orientation) of each vehicle, predicts its motion, and assigns a consistent tracking ID over time.

***

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Vision-Based-CIP.git](https://github.com/your-username/Vision-Based-CIP.git)
    cd Vision-Based-CIP
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The required packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Project Data:**
    For the pipeline to function, you must download the required data folders (`dataset`, `weights`, and `camera_data`) from the provided Google Drive link.
    
    *   **Google Drive Link:** [LINK] (Private link to be provided)

    **Folder Contents:**
    *   `dataset/`: Contains the video clips (`video_clips/`) and ground truth data (`LUMPI/LUMPI_gt.csv`) required for processing and evaluation.
    *   `weights/`: Contains the pre-trained YOLO model weights (e.g., `yolov8s_lumpi_custom.pt`, `yolo11m.pt`).
    *   `camera_data/`: Contains essential calibration files, including homography matrices and top-down satellite images for projection.

    **Action:** Unzip these folders into the root directory of the repository. Your directory structure should look like this:
    ```
    Vision-Based-CIP/
    ├── dataset/
    ├── weights/
    ├── camera_data/
    ├── multi_camera_tracking_pipeline/
    └── ...
    ```

***

## Usage

The main pipeline is executed via the command line using `multi_camera_tracking_pipeline/main.py`.

### Configuration
The pipeline configuration is managed in `multi_camera_tracking_pipeline/multi_camera_tracker/config.py`.
*   **Dataset Configuration**: The `LUMPI` and `CarLA` dictionaries define paths for videos, homography matrices, and camera ROIs.
*   **Model Paths**: Ensure `MODEL_PATH` points to the correct YOLO weights.

> [!NOTE]
> If your dataset directory structure differs from the default, you must update the paths in `config.py` before running the pipeline.

### Running the Pipeline
Run the pipeline from the root directory, specifying the dataset to use:

```bash
# For the LUMPI dataset
python multi_camera_tracking_pipeline/main.py --dataset LUMPI

# For the CarLA dataset (if configured)
python multi_camera_tracking_pipeline/main.py --dataset CarLA
```

The system will:
1.  Load the configuration for the selected dataset.
2.  Process video feeds from all configured cameras.
3.  Generate output files in the `output/` directory.
