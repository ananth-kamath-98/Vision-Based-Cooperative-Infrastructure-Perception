import cv2
import os
import numpy as np
from ultralytics import YOLO

from . import config
from .detection import detect_and_cluster_vehicles, assign_movement_direction
from .fusion import fuse_detections_across_cameras
from .tracker import Tracker, time_step, H, R, Q, MAX_MISSED_FRAMES, \
    DISTANCE_THRESHOLD
from .transformation import Camera
from .visualization import draw_vehicle_detections


def run_pipeline(dataset_config):
    """
    Initializes and runs the full multi-camera tracking pipeline.
    """
    # --- Initialization ---
    model = YOLO(config.MODEL_PATH)

    satellite_base = cv2.imread(dataset_config["satellite_image_path"])
    if satellite_base is None:
        raise FileNotFoundError(
            f"Satellite image not found at: {dataset_config['satellite_image_path']}")

    output_dir = os.path.join(config.OUTPUT_DIR, dataset_config["name"])
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'tracked_output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, config.FRAME_RATE,
                                 (satellite_base.shape[1],
                                  satellite_base.shape[0]))

    results_path = os.path.join(output_dir,
                                dataset_config.get("results_filename",
                                                   "tracking_results.txt"))
    results_file = open(results_path, "w")

    cameras = {cam_id: Camera(cam_id, **cam_info) for cam_id, cam_info in
               dataset_config["cameras"].items()}
    tracker_manager = Tracker(dt=time_step,
                              H=H,
                              R=R,
                              Q=Q,
                              max_missed_frames=MAX_MISSED_FRAMES,
                              distance_threshold=50.0)
    frame_count = 1

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
        fused_detections = fuse_detections_across_cameras(
            all_detections_for_fusion)

        # 3. Update trackers and visualize
        satellite_view = satellite_base.copy()  # Initialize satellite view once per frame
        active_tracks = tracker_manager.process_frame(fused_detections)
        for track in active_tracks:
            # Get the track's current (updated) position
            pos = np.dot(H, track.state).flatten()
            x_pos, y_pos = int(pos[0]), int(pos[1])
            track_id = track.id

            # Draw a circle for the track
            cv2.circle(satellite_view, (x_pos, y_pos), 7, (0, 255, 0), -1)

            # Draw the track ID
            cv2.putText(
                satellite_view,
                f"ID: {track_id}",
                (x_pos + 10, y_pos + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Write the final satellite view with all tracks to the video file
        out_writer.write(satellite_view)

        # 4. Save tracking results (uncomment and adapt if needed)
        for track in active_tracks:
            pos = np.dot(H, track.state).flatten()
            results_file.write(f"{frame_count},{track.id},{pos[0]},{pos[1]},-1,-1,-1,-1\n")

        # 5. Display frames
        for cam_id, frame in frames.items():
            cv2.imshow(cam_id, frame)
        cv2.imshow("Satellite View", satellite_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # --- Cleanup ---
    for cam in cameras.values():
        cam.video_capture.release()
    out_writer.release()
    results_file.close()
    cv2.destroyAllWindows()
