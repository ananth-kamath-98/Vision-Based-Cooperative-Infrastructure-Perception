import cv2
import numpy as np

class Camera:
    """
    Encapsulates all data and operations for a single camera, including
    video capture, homography transformation, and movement ROIs.
    """
    def __init__(self, cam_id, video_path, homography_path, label, color, movement_rois):
        self.cam_id = cam_id
        self.video_path = video_path
        self.homography_matrix = np.loadtxt(homography_path)
        self.label = label
        self.color = color
        self.movement_rois = movement_rois
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

    def transform_point(self, point):
        """Transforms a single point using the camera's homography matrix."""
        pt_homogeneous = np.array([[point[0], point[1], 1]], dtype='float32').T
        mapped_pt = self.homography_matrix.dot(pt_homogeneous)
        mapped_pt /= mapped_pt[2]
        return int(mapped_pt[0, 0]), int(mapped_pt[1, 0])
