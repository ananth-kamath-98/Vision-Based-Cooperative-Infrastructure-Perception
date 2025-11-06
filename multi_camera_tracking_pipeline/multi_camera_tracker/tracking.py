import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from . import config


def unwrap_angle(angle, prev_angle):
    diff = angle - prev_angle
    while diff > math.pi: diff -= 2 * math.pi
    while diff <= -math.pi: diff += 2 * math.pi
    return prev_angle + diff


class EKFTracker:
    """Extended Kalman Filter for a single object."""

    def __init__(self, initial_state, P, Q, R, dt):
        self.x = initial_state
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt
        self.missed_frames = 0
        self.lost_age = 0
        self.rolling_vx = 0.0
        self.rolling_vy = 0.0

    def predict(self):
        F = np.array([
            [1, 0, self.dt, 0, 0], [0, 1, 0, self.dt, 0], [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.x.shape[0]) - K @ H) @ self.P
        self.missed_frames = 0


class TrackerManager:
    """Manages the lifecycle of all trackers."""

    def __init__(self):
        self.active_trackers = {}
        self.lost_trackers = {}
        self.next_tracker_id = 0
        self.tracker_prev_meas = {}

    def _data_association(self, trackers, detections):
        if not trackers or not detections:
            return {}, set(trackers.keys()), set(range(len(detections)))

        tracker_ids = list(trackers.keys())
        cost_matrix = np.zeros((len(tracker_ids), len(detections)))
        for i, t_id in enumerate(tracker_ids):
            pred_state = trackers[t_id].x
            for j, det in enumerate(detections):
                cost_matrix[i, j] = math.hypot(pred_state[0, 0] - det["x"],
                                               pred_state[1, 0] - det["y"])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched, unmatched_t, unmatched_d = {}, set(tracker_ids), set(
            range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < config.ASSOCIATION_THRESHOLD:
                t_id = tracker_ids[r]
                matched[t_id] = detections[c]
                unmatched_t.discard(t_id)
                unmatched_d.discard(c)
        return matched, unmatched_t, unmatched_d

    def update_trackers(self, detections):
        for tracker in self.active_trackers.values():
            tracker.predict()

        matched, unmatched_t, unmatched_d_indices = self._data_association(
            self.active_trackers, detections)

        for t_id, det in matched.items():
            self._update_matched_tracker(t_id, det)

        for t_id in list(unmatched_t):
            tr = self.active_trackers[t_id]
            tr.missed_frames += 1
            if tr.missed_frames > config.MAX_MISSED_FRAMES:
                tr.lost_age = 0
                self.lost_trackers[t_id] = self.active_trackers.pop(t_id)

        for t_id in list(self.lost_trackers.keys()):
            self.lost_trackers[t_id].lost_age += 1
            if self.lost_trackers[t_id].lost_age > config.LOST_TRACKER_TIMEOUT:
                del self.lost_trackers[t_id]

        unmatched_detections = [detections[i] for i in unmatched_d_indices]
        self._revive_or_create_trackers(unmatched_detections)

    def _update_matched_tracker(self, tracker_id, detection):
        tracker = self.active_trackers[tracker_id]
        center = (detection["x"], detection["y"])

        if tracker_id in self.tracker_prev_meas:
            prev_meas = self.tracker_prev_meas[tracker_id]
            vx = (center[0] - prev_meas[0]) / config.DT
            vy = (center[1] - prev_meas[1]) / config.DT

            ALPHA = 0.2
            tracker.rolling_vx = ALPHA * vx + (1 - ALPHA) * tracker.rolling_vx
            tracker.rolling_vy = ALPHA * vy + (1 - ALPHA) * tracker.rolling_vy

            raw_theta = math.atan2(tracker.rolling_vy, tracker.rolling_vx)
            if math.hypot(tracker.rolling_vx, tracker.rolling_vy) < 90.0:
                raw_theta = float(tracker.x[4, 0])
            theta_meas = unwrap_angle(raw_theta, float(tracker.x[4, 0]))
        else:
            theta_meas = float(tracker.x[4, 0])

        measurement = np.array([center[0], center[1], theta_meas]).reshape(3,
                                                                           1)
        tracker.update(measurement)
        self.tracker_prev_meas[tracker_id] = center

    def _revive_or_create_trackers(self, unmatched_detections):
        for det in unmatched_detections:
            det_pt = np.array([det["x"], det["y"]])
            found_match = False

            for lost_id, lost_tr in list(self.lost_trackers.items()):
                lost_pt = np.array([lost_tr.x[0, 0], lost_tr.x[1, 0]])
                if np.linalg.norm(
                        det_pt - lost_pt) < config.ASSOCIATION_THRESHOLD:
                    theta_meas = float(lost_tr.x[4, 0])
                    measurement = np.array(
                        [det["x"], det["y"], theta_meas]).reshape(3, 1)
                    lost_tr.update(measurement)
                    self.active_trackers[lost_id] = self.lost_trackers.pop(
                        lost_id)
                    self.tracker_prev_meas[lost_id] = (det["x"], det["y"])
                    found_match = True
                    break

            if not found_match:
                self._create_new_tracker(det)

    def _create_new_tracker(self, detection):
        center = (detection["x"], detection["y"])
        init_theta = config.DIR_TO_THETA.get(detection["dir"], 0.0)
        init_state = np.array(
            [center[0], center[1], 0.0, 0.0, init_theta]).reshape(5, 1)

        new_tracker = EKFTracker(init_state, config.P_MATRIX.copy(),
                                 config.Q_MATRIX.copy(),
                                 config.R_MATRIX.copy(), config.DT)

        self.active_trackers[self.next_tracker_id] = new_tracker
        self.tracker_prev_meas[self.next_tracker_id] = center
        self.next_tracker_id += 1
