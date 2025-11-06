import cv2
import os
from ultralytics import YOLO

from . import config
from .detection import detect_and_cluster_vehicles, assign_movement_direction
from .fusion import fuse_detections_across_cameras
from .tracking import TrackerManager
from .transformation import Camera
from .visualization import draw_vehicle_detections, draw_tracked_objects

def run_pipeline(dataset_config):
    """
    Initializes and runs the full multi-camera tracking pipeline.
    """
    # --- Initialization ---
    model = YOLO(config.MODEL_PATH)

    satellite_base = cv2.imread(dataset_config["satellite_image_path"])
    if satellite_base is None:
        raise FileNotFoundError(f"Satellite image not found at: {dataset_config['satellite_image_path']}")

    output_dir = os.path.join(config.OUTPUT_DIR, dataset_config["name"])
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'tracked_output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, config.FRAME_RATE, (satellite_base.shape[1], satellite_base.shape[0]))

    cameras = {cam_id: Camera(cam_id, **cam_info) for cam_id, cam_info in dataset_config["cameras"].items()}
    tracker_manager = TrackerManager()

    # --- Main Loop ---
    while True:
        frames = {}
        all_detections_for_fusion = []

        # 1. Detect, Cluster, and Transform from all cameras
        for cam_id, cam in cameras.items():
            ret, frame = cam.video_capture.read()
            if not ret:
                break
            frames[cam_id] = frame

            clustered_points = detect_and_cluster_vehicles(model, frame)

            for pt in clustered_points:
                mapped_pt = cam.transform_point(pt)
                direction = assign_movement_direction(pt, cam.movement_rois)
                all_detections_for_fusion.append({
                    "pt": mapped_pt,
                    "cam": cam.label,
                    "dir": direction,
                })

            draw_vehicle_detections(frame, clustered_points, cam.color)

        if len(frames) < len(cameras):
            break

        # 2. Fuse detections from all cameras
        fused_detections = fuse_detections_across_cameras(all_detections_for_fusion)

        # 3. Update trackers
        tracker_manager.update_trackers(fused_detections)

        # 4. Visualization
        satellite_view = satellite_base.copy()
        draw_tracked_objects(satellite_view, tracker_manager.active_trackers)
        out_writer.write(satellite_view)

        # Display frames
        for cam_id, frame in frames.items():
            cv2.imshow(cam_id, frame)
        cv2.imshow("Satellite View", satellite_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    for cam in cameras.values():
        cam.video_capture.release()
    out_writer.release()
    cv2.destroyAllWindows()
