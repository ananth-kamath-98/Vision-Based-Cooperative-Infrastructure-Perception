import math

import cv2 as cv
import numpy as np


def draw_fixed_rotated_bbox(img, center, width, height, angle_rad=math.pi / 2,
                            color=(255, 255, 255), thickness=2):
    angle_deg = math.degrees(angle_rad) - 90
    rotated_rect = (center, (width, height), angle_deg)
    box_points = cv.boxPoints(rotated_rect)
    box_points = np.intp(box_points)
    cv.polylines(img, [box_points], isClosed=True, color=color,
                 thickness=thickness)
    arrow_length = 50
    arrow_end = (int(center[0] + arrow_length * math.cos(angle_rad)),
                 int(center[1] + arrow_length * math.sin(angle_rad)))
    cv.arrowedLine(img, center, arrow_end, (0, 0, 255), thickness,
                   tipLength=0.3)
