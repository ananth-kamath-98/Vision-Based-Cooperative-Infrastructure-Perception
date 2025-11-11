import math
import os

import numpy as np

# --- General Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# --- Model and Detection Configuration ---
MODEL_PATH = os.path.join(BASE_DIR, "../weights/yolo11m.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]
DETECTION_CONFIDENCE = 0.5
DETECTION_IOU = 0.5
DETECTION_OFFSET_FACTOR = 0.1
INTRA_CAM_CLUSTER_EPSILON = 10

# --- Tracking Parameters ---
# EKF State: [x, y, vx, vy, theta]
# Q_MATRIX = np.diag([10, 10, 50, 50, 10])
# R_MATRIX = np.diag([10, 10, 50])  # Measurement noise
# P_MATRIX = np.eye(5) * 50
#
# ASSOCIATION_THRESHOLD = 100  # Max distance (pixels) for association
# MAX_MISSED_FRAMES = 15  # Frames before a tracker is 'lost'
# LOST_TRACKER_TIMEOUT = 15  # Frames before a 'lost' tracker is deleted
FRAME_RATE = 30
# DT = 1 / FRAME_RATE
FUSION_DISTANCE_THRESHOLD = 60

DIR_TO_THETA = {
    'SW': 3 * math.pi / 4,
    'SE': math.pi / 4,
    'E': math.pi / 10,
    'NW': -3 * math.pi / 4
}

# --- Dataset-Specific Configurations ---

LUMPI_CONFIG = {
    "name": "LUMPI",
    "satellite_image_path": "../camera_data/LUMPI/top_down_view/satellite_image.png",
    "results_filename": "lumpi_tracking_results.txt",
    "cameras": {
        "cam5": {
            "video_path": "../dataset/cam/5/cam_5.mp4",
            "homography_path": "../camera_data/LUMPI/cam5/homography_matrix.txt",
            "label": "C5",
            "color": (0, 0, 255),
            "movement_rois": [
                {"direction": "SW",
                 "poly": [(626, 805), (178, 1035), (26, 875), (435, 710)]},
                {"direction": "SE",
                 "poly": [(1055, 497), (1397, 467), (1550, 560), (1162, 600)]},
                {"direction": "E",
                 "poly": [(971, 747), (1764, 1061), (1916, 850), (1149, 661)]},
            ]
        },
        "cam6": {
            "video_path": "../dataset/cam/6/cam_6.mp4",
            "homography_path": "../camera_data/LUMPI/cam6/homography_matrix.txt",
            "label": "C6",
            "color": (0, 255, 0),
            "movement_rois": [
                {"direction": "SW",
                 "poly": [(1063, 931), (1404, 1227), (1639, 963),
                          (1461, 832)]},
                {"direction": "E",
                 "poly": [(746, 484), (936, 412), (1100, 484), (918, 540)]},
                {"direction": "SE",
                 "poly": [(1206, 542), (1620, 483), (1637, 616), (1344, 662)]},
            ]
        },
        "cam7": {
            "video_path": "../dataset/cam/7/cam_7.mp4",
            "homography_path": "../camera_data/LUMPI/cam7/homography_matrix.txt",
            "label": "C7",
            "color": (255, 0, 0),
            "movement_rois": [
                {"direction": "SW",
                 "poly": [(617, 538), (405, 380), (216, 402), (318, 570)]},
                {"direction": "NW",
                 "poly": [(771, 489), (1238, 423), (1438, 462), (893, 592)]},
            ]
        }
    }
}

CARLA_CONFIG = {
    "name": "CarLA",
    "satellite_image_path": "../camera_data/CarLA/top_down_view/satellite_image.jpg",
    "results_filename": "carla_tracking_results.txt",
    "cameras": {
        "cam1": {
            "video_path": "../dataset/CarLA/Camera_1/video/Camera_1.mp4",
            "homography_path": "../camera_data/CarLA/Camera_1/homography_matrix.txt",
            # Placeholder
            "label": "C1",
            "color": (0, 0, 255),
            "movement_rois": [{"direction": "N",
                               "poly": [(180, 702), (25, 337), (189, 320),
                                        (542, 605)]},
                              {"direction": "S",
                               "poly": [(556, 603), (199, 319), (333, 310),
                                        (812, 535)]},
                              {"direction": "E",
                               "poly": [(1108, 545), (1739, 336), (1907, 363),
                                        (1370, 639)]},
                              {"direction": "W",
                               "poly": [(1346, 691), (1793, 424), (1913, 598),
                                        (1680, 823)]}]
        },
        "cam2": {
            "video_path": "../dataset/CarLA/Camera_2/video/Camera_2.mp4",
            "homography_path": "../camera_data/CarLA/Camera_2/homography_matrix.txt",
            # Placeholder
            "label": "C2",
            "color": (0, 255, 0),
            "movement_rois": [{"direction": "N",
                               "poly": [(1330, 899), (1860, 591), (1917, 847),
                                        (1743, 1069)]},
                              {"direction": "S",
                               "poly": [(1106, 753), (1661, 564), (1828, 599),
                                        (1360, 845)]},
                              {"direction": "E",
                               "poly": [(125, 924), (21, 522), (172, 507),
                                        (521, 818)]},
                              {"direction": "W",
                               "poly": [(573, 860), (212, 535), (345, 514),
                                        (882, 763)]}]

        },
        "cam3": {
            "video_path": "../dataset/CarLA/Camera_3/video/Camera_3.mp4",
            "homography_path": "../camera_data/CarLA/Camera_3/homography_matrix.txt",
            # Placeholder
            "label": "C3",
            "color": (255, 0, 0),
            "movement_rois": [{"direction": "N",
                               "poly": [(1117, 736), (1661, 518), (1809, 532),
                                        (1375, 810)]},
                              {"direction": "S",
                               "poly": [(1353, 859), (1817, 533), (1919, 545),
                                        (1845, 1025)]},
                              {"direction": "E",
                               "poly": [(591, 861), (76, 504), (215, 493),
                                        (906, 749)]},
                              {"direction": "W",
                               "poly": [(142, 941), (4, 704), (65, 507),
                                        (539, 819)]}]
        }
    }
}


def get_dataset_config(name):
    """Returns the configuration dictionary for the specified dataset."""
    if name.upper() == 'LUMPI':
        return LUMPI_CONFIG
    elif name.upper() == 'CARLA':
        print(
            "Warning: CarLA dataset paths and ROIs are placeholders. Please update config.py.")
        return CARLA_CONFIG
    else:
        raise ValueError(f"Unknown dataset: {name}")
