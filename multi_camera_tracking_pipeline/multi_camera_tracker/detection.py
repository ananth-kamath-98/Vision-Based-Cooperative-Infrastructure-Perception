import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from . import config


def detect_and_cluster_vehicles(model, frame):
    """
    Performs vehicle detection and finds their bottom-center points.
    Returns a list of centroid points for a single camera view.
    """
    results = model(frame, conf=config.DETECTION_CONFIDENCE,
                    iou=config.DETECTION_IOU, classes=config.VEHICLE_CLASSES,
                    verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            offset = int((y2 - y1) * config.DETECTION_OFFSET_FACTOR)
            bottom_center = ((x1 + x2) // 2 - offset, y2 - offset)
            detections.append(bottom_center)

    if not detections:
        return []

    # Cluster raw detections using DBSCAN
    points = np.array(detections)
    db = DBSCAN(eps=config.INTRA_CAM_CLUSTER_EPSILON, min_samples=1).fit(
        points)

    fused_points = []
    for label in set(db.labels_):
        cluster_points = points[db.labels_ == label]
        centroid = np.mean(cluster_points, axis=0)
        fused_points.append(tuple(map(int, centroid)))

    return fused_points


def assign_movement_direction(point, movement_rois):
    """
    Assigns a movement direction to a point based on predefined ROIs.
    """
    for roi in movement_rois:
        poly = np.array(roi["poly"], dtype=np.int32)
        if cv2.pointPolygonTest(poly, point, False) >= 0:
            return roi["direction"]
    return None
