import numpy as np
import cv2 as cv


class MapAlignmentProcessor:
    def __init__(self, target_image_path, other_image_path,
                 cam_target_points_path, cam_other_points_path, color,
                 cam_label):
        self.target_image_path = target_image_path
        self.other_image_path = other_image_path
        self.target_img = cv.imread(self.target_image_path)
        if self.target_img is None:
            raise ValueError("Error: Could not load target (satellite) image.")
        self.target_img_original = self.target_img.copy()
        pts_target = np.array(self.read_points(cam_target_points_path),
                              dtype=np.float32)
        pts_other = np.array(self.read_points(cam_other_points_path),
                             dtype=np.float32)
        self.H, _ = cv.findHomography(pts_other, pts_target, cv.RANSAC)
        if self.H is None:
            raise ValueError("Error: Homography computation failed.")
        self.color = color
        self.cam_label = cam_label

    def read_points(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        points = []
        for line in lines:
            point = [float(x) for x in line.strip().split()]
            points.append(point)
        return points

    def transform_point(self, point):
        pt = np.array([[point[0], point[1], 1]], dtype='float32').T
        mapped = self.H.dot(pt)
        mapped /= mapped[2]
        return (int(mapped[0, 0]), int(mapped[1, 0]))

    def reset_display(self):
        self.target_img_display = self.target_img_original.copy()
