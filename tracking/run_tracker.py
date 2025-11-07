import cv2
import numpy as np
from ultralytics import YOLO

# Import our tracker classes
# (Make sure tracker.py and track.py are in the same directory)
from tracker import Tracker, H, R, Q, time_step, MAX_MISSED_FRAMES, DISTANCE_THRESHOLD

# --- Configuration ---
VIDEO_PATH = "C:\\Users\\agk98\\Desktop\\repos\\Vision-Based-Cooperative-Infrastructure-Perception\\dataset\\CarLA\\Camera_1\\video\\Camera_1.mp4"       # <<< SET YOUR VIDEO PATH
YOLO_MODEL_PATH = "C:\\Users\\agk98\\Desktop\\repos\\Vision-Based-Cooperative-Infrastructure-Perception\\weights\\yolo11m_custom_300e.pt"   # <<< SET YOUR YOLO MODEL PATH
OUTPUT_VIDEO_PATH = "output_video.mp4"

# (Optional) We only want to track 'car', 'truck', 'bus'
CLASSES_TO_TRACK = [0] # COCO class IDs for car, bus, truck

# --- Main Application ---
def main():
    # 1. Load YOLO Model
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)

    # 2. Initialize Tracker
    print("Initializing Tracker...")
    tracker = Tracker(
        dt=time_step,
        H=H,
        R=R,
        Q=Q,
        max_missed_frames=MAX_MISSED_FRAMES,
        distance_threshold=DISTANCE_THRESHOLD
    )

    # 3. Load Video
    print(f"Loading video from {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # 4. Get Video Properties for Output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 5. Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Saving output video to {OUTPUT_VIDEO_PATH}...")

    frame_count = 0

    # 6. --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # 7. Run YOLO Detection
        # We run detection and filter for our desired classes
        results = model(frame, conf=0.9, iou=0.5, classes=CLASSES_TO_TRACK, verbose=False, device=0)

        # 8. Format Detections
        # Our tracker expects a list of [x, y] centroids
        detections = []
        for box in results[0].boxes:
            # Get center_x, center_y
            x_center, y_center, w, h = box.xywh[0]
            detections.append([float(x_center), float(y_center)])

        # 9. --- Call the Tracker ---
        # This is the single most important line:
        # It runs all 5 steps: Predict, Associate, Update, Manage, Clean Up
        active_tracks = tracker.process_frame(detections)

        # 10. Draw Results
        for track in active_tracks:
            # Get the track's *current* (updated) position
            # We use H to select [px, py] from the [px, py, v, theta, theta_dot] state
            pos = np.dot(H, track.state).flatten()
            x_pos, y_pos = int(pos[0]), int(pos[1])
            track_id = track.id

            # Draw a circle for the track
            cv2.circle(frame, (x_pos, y_pos), 7, (0, 255, 0), -1) # Green circle

            # Draw the track ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x_pos + 10, y_pos + 10), # Position offset from circle
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, # Font scale
                (0, 255, 0), # Color (Green)
                2 # Thickness
            )

        # (Optional) Draw detections as red 'x'
        # for det in detections:
        #     x_det, y_det = int(det[0]), int(det[1])
        #     cv2.line(frame, (x_det - 5, y_det - 5), (x_det + 5, y_det + 5), (0, 0, 255), 2)
        #     cv2.line(frame, (x_det + 5, y_det - 5), (x_det - 5, y_det + 5), (0, 0, 255), 2)


        # 11. Write Frame to Output Video
        out.write(frame)

        # (Optional) Display the video live
        # cv2.imshow("EKF Tracker", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 12. Release Resources
    print("Processing complete. Releasing resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()