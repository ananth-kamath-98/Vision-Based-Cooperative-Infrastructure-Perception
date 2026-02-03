# python
import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from . import config
from .detection import detect_and_cluster_vehicles, assign_movement_direction
from .fusion import fuse_detections_across_cameras
from .tracker import Tracker, time_step, H, R, Q, MAX_MISSED_FRAMES
from .transformation import Camera
from .visualization import draw_vehicle_detections


def run_pipeline(dataset_config):
    """
    Runs the multi-camera tracking pipeline for multiple YOLO models.
    Each model's outputs are stored in a folder named after the model under
    `config.OUTPUT_DIR/{dataset_name}/{model_name}`. Each model is run for up to
    1000 frames (or until the videos end).
    """
    model_names = [
        # "yolov8s",
        # "yolo11m_custom",
        # "yolo11m",
        # "yolov8s_custom_300e",
        # "yolov7-seg",
        # "yolov7",
        "yolov8s_lumpi_custom"
    ]

    weights_dir = os.path.join("..", "weights")

    for model_name in model_names:
        # locate weight file
        candidate_paths = [
            os.path.join(weights_dir, model_name),
            os.path.join(weights_dir, model_name + ".pt"),
        ]
        weight_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                weight_path = p
                break
        if weight_path is None:
            # fallback: find any file that starts with model_name
            for fname in os.listdir(weights_dir) if os.path.exists(weights_dir) else []:
                if fname.startswith(model_name):
                    weight_path = os.path.join(weights_dir, fname)
                    break

        if weight_path is None:
            print(f"Weight for model {model_name} not found in {weights_dir}, skipping.")
            continue

        print(f"Running pipeline with model {model_name} ({weight_path})")

        # init model
        model = YOLO(weight_path)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # create output folders per model
        model_output_dir = os.path.join(config.OUTPUT_DIR, dataset_config["name"], model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # evaluation folder
        eval_dir = os.path.join("evaluation", dataset_config["name"], model_name)
        os.makedirs(eval_dir, exist_ok=True)

        # prepare video writer and satellite image
        satellite_base = cv2.imread(dataset_config["satellite_image_path"])
        if satellite_base is None:
            raise FileNotFoundError(
                f"Satellite image not found at: {dataset_config['satellite_image_path']}"
            )

        output_video_path = os.path.join(model_output_dir, "tracked_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            output_video_path, fourcc, config.FRAME_RATE, (satellite_base.shape[1], satellite_base.shape[0])
        )

        results_path = os.path.join(model_output_dir, dataset_config.get("results_filename", f"{model_name}_tracking_results.txt"))
        results_file = open(results_path, "w")

        # detected centroids per model
        detected_centroids_path = os.path.join(eval_dir, "fusion.csv")
        detected_centroids_file = open(detected_centroids_path, "w")
        detected_centroids_file.write("frame,centroid_x,centroid_y\n")

        # initialize cameras and tracker fresh for each model
        cameras = {cam_id: Camera(cam_id, **cam_info) for cam_id, cam_info in dataset_config["cameras"].items()}
        tracker_manager = Tracker(
            dt=time_step, H=H, R=R, Q=Q, max_missed_frames=MAX_MISSED_FRAMES, distance_threshold=50.0
        )

        frame_count = 1
        max_frames = 1000
        latency_data = []

        while frame_count <= max_frames:
            frame_start_time = time.time()
            print(f"[{model_name}] Processing frame {frame_count}...")
            frames = {}
            all_detections_for_fusion = []
            total_detection_latency = 0.0

            # 1. Detect, Cluster, and Transform from all cameras
            for cam_id, cam in cameras.items():
                ret, frame = cam.video_capture.read()
                if not ret:
                    break
                frames[cam_id] = frame

                detection_start_time = time.time()
                clustered_points = detect_and_cluster_vehicles(model, frame)
                detection_end_time = time.time()
                total_detection_latency += (detection_end_time - detection_start_time)

                for pt in clustered_points:
                    mapped_pt = cam.transform_point(pt)
                    direction = assign_movement_direction(pt, cam.movement_rois)
                    all_detections_for_fusion.append({"pt": mapped_pt, "cam": cam.label, "dir": direction})

            if len(frames) < len(cameras):
                print(f"[{model_name}] Video ended or missing frames; stopping model run.")
                break

            # 2. Fuse detections from all cameras
            fusion_start_time = time.time()
            fused_detections = fuse_detections_across_cameras(all_detections_for_fusion)
            fusion_end_time = time.time()
            fusion_latency = fusion_end_time - fusion_start_time

            for det in fused_detections:
                detected_centroids_file.write(f"{frame_count},{det['x']},{det['y']}\n")

            # 3. Update trackers and visualize
            tracking_start_time = time.time()
            active_tracks = tracker_manager.process_frame(fused_detections)
            tracking_end_time = time.time()
            tracking_latency = tracking_end_time - tracking_start_time

            frame_end_time = time.time()
            end_to_end_latency = frame_end_time - frame_start_time

            latency_data.append(
                {
                    "frame": frame_count,
                    "detection_latency_ms": total_detection_latency * 1000,
                    "fusion_latency_ms": fusion_latency * 1000,
                    "tracking_latency_ms": tracking_latency * 1000,
                    "end_to_end_latency_ms": end_to_end_latency * 1000,
                }
            )
            print(f"[{model_name}] Frame {frame_count} processed in {end_to_end_latency:.4f} seconds.")

            # 4. Save tracking results
            for track in active_tracks:
                pos = np.dot(H, track.state).flatten()
                results_file.write(f"{frame_count},{track.id},{pos[0]},{pos[1]},-1,-1,-1,-1\n")

            # 5. (Optional) write satellite view video if needed
            # out_writer.write(satellite_view)  # uncomment when drawing is enabled

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

        # --- Cleanup for this model ---
        for cam in cameras.values():
            try:
                cam.video_capture.release()
            except Exception:
                pass
        out_writer.release()
        results_file.close()
        detected_centroids_file.close()
        cv2.destroyAllWindows()

        # save latency report for this model
        latency_df = pd.DataFrame(latency_data)
        latency_output_path = os.path.join(model_output_dir, "latency_report.csv")
        latency_df.to_csv(latency_output_path, index=False)
        print(f"[{model_name}] Latency report saved to {latency_output_path}")

    print("All model runs complete.")