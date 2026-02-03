import cv2
import math

def draw_vehicle_detections(frame, clustered_points, color):
    """Draws clustered detection points on a camera frame."""
    for pt in clustered_points:
        cv2.circle(frame, pt, 6, color, -1)
    return frame

def draw_tracked_objects(image, trackers):
    """Draws tracked objects with their ID and orientation on the satellite view."""
    for tracker_id, tracker in trackers.items():
        state = tracker.x
        pos = (int(state[0, 0]), int(state[1, 0]))
        theta = state[4, 0]

        # Draw a circle for the object's position
        cv2.circle(image, pos, 8, (255, 255, 255), -1)

        # Draw orientation arrow
        arrow_length = 30
        arrow_end = (
            int(pos[0] + arrow_length * math.cos(theta)),
            int(pos[1] + arrow_length * math.sin(theta))
        )
        cv2.arrowedLine(image, pos, arrow_end, (0, 0, 255), 2, tipLength=0.4)

        # Draw tracker ID
        cv2.putText(image, str(tracker_id), (pos[0] + 10, pos[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
