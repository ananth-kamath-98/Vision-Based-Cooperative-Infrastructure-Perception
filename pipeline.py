import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from tracker import initial_position, unwrap_angle, EKFTracker
from utils import draw_fixed_rotated_bbox


class Pipeline:
    class CamDetails:
        def __init__(self, cam_id, color, cam_label):
            self.cam_id = cam_id
            self.video = str(
                Path(
                    __file__).parent / "dataset" / "cam" / f"{cam_id}" / f"cam_{cam_id}.mp4")
            self.homography_matrix = np.loadtxt(Path(
                __file__).parent / "camera_data" / f"cam{cam_id}" / "homography_matrix.txt")
            self.color = color
            self.cam_label = cam_label

            self.movement_matrix = None
            self.assign_movement(cam_id)

        def assign_movement(self, cam_id):
            if cam_id == 5:
                self.movement_matrix = [{"direction": "SW",
                                         "poly": [(626, 805), (178, 1035),
                                                  (26, 875), (435, 710)]},
                                        {"direction": "SE",
                                         "poly": [(971, 747), (1764, 1061),
                                                  (1916, 850), (1149, 661)]},
                                        {"direction": "E",
                                         "poly": [(1055, 497), (1397, 467),
                                                  (1421, 514), (1136, 544)]},
                                        {"direction": "NW",
                                         "poly": [(302, 674), (588, 558),
                                                  (182, 504), (10, 556)]}]
            elif cam_id == 6:
                self.movement_matrix = [{"direction": "SW",
                                         "poly": [(1063, 931), (1404, 1227),
                                                  (1639, 963), (1461, 832)]},
                                        {"direction": "SE",
                                         "poly": [(1206, 542), (1620, 483),
                                                  (1637, 616), (1344, 662)]},
                                        {"direction": "E",
                                         "poly": [(746, 484), (936, 412),
                                                  (1099, 423), (957, 491)]},
                                        {"direction": "NW",
                                         "poly": [(722, 819), (4, 1043),
                                                  (7, 765), (556, 623)]}]
            else:
                self.movement_matrix = [{"direction": "SW",
                                         "poly": [(617, 538), (405, 380),
                                                  (216, 402), (318, 570)]},
                                        {"direction": "NW",
                                         "poly": [(771, 489), (1238, 423),
                                                  (1438, 462), (893, 592)]}]

        def transform_point(self, point):
            pt = np.array([[point[0], point[1], 1]], dtype='float32').T
            mapped = self.homography_matrix.dot(pt)
            mapped /= mapped[2]
            return int(mapped[0, 0]), int(mapped[1, 0])

    class Checkpoint:
        def __init__(self):
            self.frame1 = None
            self.frame2 = None
            self.frame3 = None

        def show_current_checkpoint(self, satellite_view):
            if self.frame1 is not None:
                cv2.imshow("Camera 5", self.frame1)
            if self.frame2 is not None:
                cv2.imshow("Camera 6", self.frame2)
            if self.frame3 is not None:
                cv2.imshow("Camera 7", self.frame3)
            cv2.imshow("Satellite View", satellite_view)

    def __init__(self):
        # load all the media (vids + images) and homography matrices
        self.model = YOLO(model="yolov8s.pt")

        # load camera data
        self.cam5 = self.CamDetails(cam_id=5, color=(0, 0, 255),
                                    cam_label="C5")
        self.cam6 = self.CamDetails(cam_id=6, color=(0, 255, 0),
                                    cam_label="C6")
        self.cam7 = self.CamDetails(cam_id=7, color=(255, 0, 0),
                                    cam_label="C7")

        # load satellite data
        self.satellite_image = cv2.imread(str(Path(
            __file__).parent / "camera_data" / "top_down_view" / "satellite_image.png"))

        self.detected_dict = dict()

        self.check_point = None
        self.cap5 = None
        self.cap6 = None
        self.cap7 = None

        self.active_trackers = {}

        self.AVG_CAR_WIDTH = 50
        self.AVG_CAR_HEIGHT = 100

        self.dt = 1 / 30.0
        self.P = np.eye(5) * 1.0
        self.Q = np.eye(5)
        self.Q[0, 0] = self.Q[1, 1] = 0.2
        self.Q[2, 2] = self.Q[3, 3] = 0.05
        self.Q[4, 4] = 0.01
        self.R = np.eye(3)
        self.R[0, 0] = self.R[1, 1] = 0.5
        self.R[2, 2] = 0.1

        self.check_point = self.Checkpoint()

        self.DIR_TO_THETA = {
            'SW': 3 * math.pi / 4,
            'SE': math.pi / 4,
            'E': math.pi / 10,
            'NW': -3 * math.pi / 4
        }

        self.next_tracker_id = 0

        # map tracker_id → last (x,y) measurement
        self.tracker_prev_meas = {}

        # pool of “lost” trackers
        self.lost_trackers = {}

        # how many frames to let a tracker miss before moving it to lost
        self.max_missed_frames = 5

        # how many frames to keep a lost tracker around for possible revival
        self.lost_tracker_timeout = 10

        # distance threshold for reviving a lost tracker
        self.association_threshold = 60

    def detect_and_get_points(self, cap, color, offset_factor=0.1):
        ret, frame = cap.read()
        if not ret:
            return None, None

        results = self.model([frame], conf=0.5, iou=0.5, classes=[2, 7])
        points = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0]
                x1, y1, x2, y2 = map(int, xyxy)
                offset = int((y2 - y1) * offset_factor)
                bottom_center = (((x1 + x2) // 2) - offset, y2 - offset)
                points.append(bottom_center)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame,
                            f"{self.model.names[int(box.cls[0])]}: {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (bottom_center[0], bottom_center[1]), 6,
                           color, -1)
            return frame, points

    def bbox_cluster_per_camera(self, detections, epsilon=10, min_samples=1):
        if len(detections) == 0:
            return []
        X = np.array(detections)
        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
        labels = db.labels_
        unique_labels = set(labels)
        fused_points = []
        for label in unique_labels:
            if label == -1:
                noise_points = X[labels == -1]
                for pt in noise_points:
                    fused_points.append(tuple(pt))
            else:
                cluster = X[labels == label]
                centroid = np.mean(cluster, axis=0)
                fused_points.append((int(centroid[0]), int(centroid[1])))
        return fused_points

    def assign_movement(self, point, movement_rois):
        for m in movement_rois:
            if cv2.pointPolygonTest(np.array(m["poly"], dtype=np.int32), point,
                                    False) >= 0:
                return m["direction"]
        return None

    def perform_object_detection(self):
        points_and_cam = []
        self.check_point.frame1, points1 = self.detect_and_get_points(
            self.cap5,
            color=(0, 0, 255),
            offset_factor=0.1)
        points_and_cam.append((points1, self.cam5))

        self.check_point.frame2, points2 = self.detect_and_get_points(
            self.cap6,
            color=(0, 255, 0),
            offset_factor=0.1)
        points_and_cam.append((points2, self.cam6))

        self.check_point.frame3, points3 = self.detect_and_get_points(
            self.cap7,
            color=(0, 255, 255),
            offset_factor=0.1)
        points_and_cam.append((points3, self.cam7))

        all_detections = []
        for pts, cam in points_and_cam:
            if not pts:
                continue
            for pt in self.bbox_cluster_per_camera(pts):
                mapped_pt = cam.transform_point(pt)
                all_detections.append({
                    "pt": mapped_pt,
                    "cam": cam.cam_label,
                    "dir": self.assign_movement(pt, cam.movement_matrix),
                })

        return all_detections

    def fuse_points_one_per_camera(self, detections, distance_threshold):
        clusters = []
        for det in detections:
            pt = np.array(det["pt"], dtype=np.float32)
            placed = False
            for cluster in clusters:
                pts = np.array([c["pt"] for c in cluster], dtype=np.float32)
                centroid = pts.mean(axis=0)
                if np.linalg.norm(pt - centroid) < distance_threshold:
                    cluster.append(det)
                    placed = True
                    break
            if not placed:
                clusters.append([det])

        fused = []
        for cluster in clusters:
            pts = np.array([c["pt"] for c in cluster], dtype=np.float32)
            cx, cy = pts.mean(axis=0).astype(int)
            cams = ",".join(sorted({c["cam"] for c in cluster}))
            dirs = [c["dir"] for c in cluster if c["dir"]]
            dir_major = max(set(dirs), key=dirs.count) if dirs else None
            fused.append({"x": cx, "y": cy, "cam": cams, "dir": dir_major})
        return fused

    def perform_fusion(self, detections):
        fused_detections = self.fuse_points_one_per_camera(detections, 60)

        detection_dicts = []
        for d in fused_detections:
            px, py = d["x"], d["y"]

            detection_dicts.append({
                "x": px,
                "y": py,
                "dir": d["dir"],
                "cam": d["cam"]
            })

        return detection_dicts

    def data_association(self, detections, cost_threshold=50):
        tracker_ids = list(self.active_trackers.keys())
        num_trackers = len(tracker_ids)
        num_detections = len(detections)
        if num_trackers == 0 or num_detections == 0:
            return {}, set(tracker_ids), set(range(num_detections))

        cost_matrix = np.zeros((num_trackers, num_detections),
                               dtype=np.float32)
        for i, t_id in enumerate(tracker_ids):
            pred_state = self.active_trackers[
                t_id].x
            pred_x, pred_y = pred_state[0, 0], pred_state[1, 0]
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = math.hypot(pred_x - detection["x"],
                                               pred_y - detection["y"])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = {}
        unmatched_tracker_ids = set(tracker_ids)
        unmatched_detection_idxs = set(range(num_detections))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < cost_threshold:
                t_id = tracker_ids[r]
                matched[t_id] = detections[c]
                unmatched_tracker_ids.discard(t_id)
                unmatched_detection_idxs.discard(c)
        return matched, unmatched_tracker_ids, unmatched_detection_idxs

    def track(self, detections, sat_view):
        # 1) Predict existing tracks
        for tracker in self.active_trackers.values():
            tracker.predict()

        # 2) Associate detections to trackers
        matched, unmatched_tracker_ids, unmatched_detection_idxs = \
            self.data_association(detections, cost_threshold=60)

        # 3) Update matched trackers
        for tracker_id, detection in matched.items():
            fused_center = (detection["x"], detection["y"])

            # Get the current state from the tracker
            tracker = self.active_trackers[tracker_id]
            tracker_state = tracker.x

            # Compute a smoothed velocity & orientation
            if tracker_id in self.tracker_prev_meas:
                prev_meas = self.tracker_prev_meas[tracker_id]
                vx, vy = initial_position(prev_meas, fused_center, self.dt)
                ALPHA = 0.2
                tracker.rolling_vx = ALPHA * vx + (
                        1 - ALPHA) * tracker.rolling_vx
                tracker.rolling_vy = ALPHA * vy + (
                        1 - ALPHA) * tracker.rolling_vy

                raw_theta = math.atan2(tracker.rolling_vy, tracker.rolling_vx)
                speed = math.hypot(tracker.rolling_vx, tracker.rolling_vy)
                if speed < 90.0:
                    raw_theta = float(tracker_state[4, 0])
                theta_meas = unwrap_angle(raw_theta,
                                          float(tracker_state[4, 0]))
            else:
                theta_meas = float(tracker_state[4, 0])

            # Form measurement and update EKF
            measurement = np.array([fused_center[0],
                                    fused_center[1],
                                    theta_meas]).reshape(3, 1)
            tracker.update(measurement)
            self.tracker_prev_meas[tracker_id] = fused_center

        # 4) Age and move unmatched active trackers to the lost pool
        for tracker_id in list(unmatched_tracker_ids):
            tr = self.active_trackers[tracker_id]
            tr.missed_frames += 1
            if tr.missed_frames > self.max_missed_frames:
                tr.lost_age = 0
                self.lost_trackers[tracker_id] = tr
                del self.active_trackers[tracker_id]

        # 5) Increment lost_age and remove expired lost trackers
        for tracker_id in list(self.lost_trackers.keys()):
            self.lost_trackers[tracker_id].lost_age += 1
            if self.lost_trackers[
                tracker_id].lost_age > self.lost_tracker_timeout:
                del self.lost_trackers[tracker_id]

        # 6) For each unmatched detection: try to revive a lost tracker, else spawn new
        for idx in list(unmatched_detection_idxs):
            det = detections[idx]
            detection_pt = np.array([det["x"], det["y"]])
            found_match = False

            # Attempt to revive
            for lost_id, lost_tr in list(self.lost_trackers.items()):
                last_state = lost_tr.x
                lost_pt = np.array([last_state[0, 0], last_state[1, 0]])
                if np.linalg.norm(
                        detection_pt - lost_pt) < self.association_threshold:
                    theta_meas = float(lost_tr.x[4, 0])
                    measurement = np.array(
                        [det["x"], det["y"], theta_meas]).reshape(3, 1)
                    lost_tr.update(measurement)
                    self.active_trackers[lost_id] = lost_tr
                    self.tracker_prev_meas[lost_id] = (det["x"], det["y"])
                    del self.lost_trackers[lost_id]
                    found_match = True
                    break

            if not found_match:
                # Spawn brand-new tracker
                fused_center = (det["x"], det["y"])
                init_theta = self.DIR_TO_THETA.get(det["dir"], 0.0)
                init_state = np.array([fused_center[0],
                                       fused_center[1],
                                       0.0, 0.0,
                                       init_theta]).reshape(5, 1)
                new_tr = EKFTracker(init_state,
                                    self.P.copy(),
                                    self.Q.copy(),
                                    self.R.copy(),
                                    self.dt)
                self.active_trackers[self.next_tracker_id] = new_tr
                self.tracker_prev_meas[self.next_tracker_id] = fused_center
                self.next_tracker_id += 1

        # 7) Draw all active tracks on the satellite view
        for tid, tr in self.active_trackers.items():
            state = tr.x
            pos = (int(state[0, 0]), int(state[1, 0]))
            theta = state[4, 0]
            draw_fixed_rotated_bbox(sat_view,
                                    pos,
                                    self.AVG_CAR_WIDTH,
                                    self.AVG_CAR_HEIGHT,
                                    theta)
            cv2.circle(sat_view, pos, 8, (255, 255, 255), -1)
            cv2.putText(sat_view,
                        str(tid),
                        (pos[0] + 5, pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)

    def open_video_streams(self):
        self.cap5 = cv2.VideoCapture(self.cam5.video)
        self.cap6 = cv2.VideoCapture(self.cam6.video)
        self.cap7 = cv2.VideoCapture(self.cam7.video)

    def release_resources(self):
        self.cap5.release()
        self.cap6.release()
        self.cap7.release()
        cv2.destroyAllWindows()

    def perform_operations(self):

        self.open_video_streams()

        # begin pipeline
        while True:
            sat_view = self.satellite_image.copy()

            # 1. Detect all vehicles and their finalized bounding boxes
            detections = self.perform_object_detection()

            # 2. Fusion
            detection_dicts = self.perform_fusion(detections)

            # Tracking
            self.track(detection_dicts, sat_view)

            # Show checkpoint
            self.check_point.show_current_checkpoint(sat_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()


if __name__ == '__main__':
    Pipeline().perform_operations()
