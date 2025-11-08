import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from track import EKFTrack

time_step = 1.0

# Measurement Matrix
H = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0]]).astype(float)

# Measurement Noise Covariance Matrix
# (Placeholder: 5-pixel std deviation in both x and y)
R_val = 25.0
R = np.array([[R_val, 0],
              [0, R_val]]).astype(float)

# Process Noise Covariance Matrix
Q_val_pos = 0.1       # Noise for px, py (let's keep this low)
Q_val_v = 10.0        # INCREASED: More noise for velocity
Q_val_theta = 0.1     # Noise for theta (let's keep this low)
Q_val_theta_dot = 10.0  # INCREASED: More noise for turn rate

Q = np.diag([
    Q_val_pos,
    Q_val_pos,
    Q_val_v,
    Q_val_theta,
    Q_val_theta_dot
]).astype(float)

# Track Deletion Threshold
MAX_MISSED_FRAMES = 10

DISTANCE_THRESHOLD = 100.0


class Tracker:
    """
     Manages all active tracks and the data association process
    """

    def __init__(self, dt, H, R, Q, max_missed_frames, distance_threshold):
        """Initializes the tracker with EKF parameters."""
        self.dt = dt
        self.H = H
        self.R = R
        self.Q = Q
        self.MAX_MISSED_FRAMES = max_missed_frames
        self.DISTANCE_THRESHOLD = distance_threshold

        self.tracks = []

    def process_frame(self, detections):
        """
        This is the main function that runs the 5-step loop
        for every new frame of detections.
        """
        # 1. Predict
        for track in self.tracks:
            track.predict(self.Q)

        # 2. Get predicted positions for all tracks
        if not self.tracks:
            predicted_positions = []
        else:
            predicted_positions = [np.dot(self.H, track.state).flatten()
                                   for track in self.tracks]

        # 3. Associate detections to tracks
        # A. Compute cost matrix
        detection_positions = np.array(detections)

        if not predicted_positions or not detections:
            cost_matrix = np.array([[]])
        else:
            # This is the core calculation:
            # cost_matrix[i, j] = distance(track_i, detection_j)
            cost_matrix = cdist(predicted_positions, detection_positions)

        # B. Run the Hungarian algorithm
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([]), np.array([])

        # 4. Update and Manage
        # A. Update matched tracks
        matched_track_indices = set()
        matched_detection_indices = set()
        for track_idx, detection_idx in zip(row_ind, col_ind):
            if cost_matrix[track_idx, detection_idx] > DISTANCE_THRESHOLD:
                continue
            self.tracks[track_idx].update(detections[detection_idx], self.H,
                                          self.R)
            self.tracks[track_idx].missed_frames = 0
            matched_track_indices.add(track_idx)
            matched_detection_indices.add(detection_idx)

        # B. Handle unmatched tracks
        for track_idx in range(len(self.tracks)):
            if track_idx not in matched_track_indices:
                self.tracks[track_idx].missed_frames += 1

        # C. Create new tracks for unmatched detections
        for detection_idx in range(len(detections)):
            if detection_idx not in matched_detection_indices:
                new_track = EKFTrack(detections[detection_idx])
                self.tracks.append(new_track)

        # D. Delete stale tracks
        self.tracks = [track for track in self.tracks
                       if track.missed_frames <= MAX_MISSED_FRAMES]

        # Return active tracks
        return self.tracks
