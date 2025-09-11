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

4.  **Download the LUMPI Dataset:**
    Download the dataset from the official source (e.g., `Measurement6/cam.zip.001`).
    * **Direct Link:** [https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/Measurement6/cam.zip.001](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/Measurement6/cam.zip.001)
    * Extract the multi-part `.zip` files (using a tool like 7-Zip) into the `dataset/` directory.

5.  **Prepare Video Clips:**
    This solution is tested on 10-second clips from each camera perspective. To generate these clips, run all the cells in the `dataset_setup/video_clippings.ipynb` notebook. This will create new video files in the `dataset/video_clips/` directory.

6.  **Download YOLOv8s Weights:**
    The project uses the `yolov8s.pt` model weights. If you don't have them, download them from the official Ultralytics release page.
    * **Link:** [YOLOv8s weights](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)
    * Place the downloaded `yolov8s.pt` file in the root directory of the project.

7.  **Extract Individual Frames (Optional):**
    If you need to generate a custom dataset for evaluation or fine-tuning, you can extract individual frames from the videos. Run the `dataset_setup/extracting_frames.ipynb` notebook.

***

## Usage

The main pipeline is run from the `pipeline.ipynb` Jupyter Notebook.

### Configuration
Before running, you **must** update the file paths inside the `pipeline.ipynb` notebook to match your local setup. This includes the paths for:
* YOLO model (`model_path`)
* Satellite image (`satellite_image_path`)
* Video clips (`cam5_video`, `cam6_video`, `cam7_video`)
* Homography point files for each camera

### Running the Pipeline
1.  Launch Jupyter Notebook or JupyterLab.
2.  Open `pipeline.ipynb`.
3.  After verifying the file paths, run all cells sequentially to start the tracking visualization.

***

##  Future Work

This project is a foundational implementation with several key areas for improvement:

-   **Evaluation Pipeline**: Develop a robust evaluation framework to quantitatively measure the performance of each component using standard metrics (mAP for detection, Localization Error for projection, and CLEAR MOT for tracking).