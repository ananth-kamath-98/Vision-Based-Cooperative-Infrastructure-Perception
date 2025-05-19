import math

import numpy as np


def process_model(x, dt):
    pos_x, pos_y, vx, vy, theta = x.ravel()
    new_x = pos_x + vx * dt
    new_y = pos_y + vy * dt
    return np.array([new_x, new_y, vx, vy, theta]).reshape(5, 1)


def measurement_model(x):
    return np.array([x[0, 0], x[1, 0], x[4, 0]]).reshape(3, 1)


def f_jacobian(x, dt):
    F = np.eye(5)
    F[0, 2] = dt
    F[1, 3] = dt
    return F

def initial_orientation(prev_pos, curr_pos, epsilon=1e-6):
    dx = curr_pos[ 0 ] - prev_pos[ 0 ]
    dy = curr_pos[ 1 ] - prev_pos[ 1 ]
    if abs(dx) < epsilon and abs(dy) < epsilon:
        return 0.0
    return math.atan2(dy, dx)


def initial_position(prev_pos, curr_pos, dt):
    dx = curr_pos[ 0 ] - prev_pos[ 0 ]
    dy = curr_pos[ 1 ] - prev_pos[ 1 ]
    vx = dx / dt
    vy = dy / dt
    return vx, vy


def unwrap_angle(angle, prev_angle):
    diff = angle - prev_angle
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff <= -math.pi:
        diff += 2 * math.pi
    return prev_angle + diff

class EKFTracker:
    def __init__(self, initial_state, P, Q, R, dt):
        self.x = initial_state
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt
        self.rolling_vx = 0.0
        self.rolling_vy = 0.0
        self.missed_frames = 0
        self.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])

    def predict(self):
        self.x = process_model(self.x, self.dt)
        F = f_jacobian(self.x, self.dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z_pred = measurement_model(self.x)
        y = z - z_pred
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.missed_frames = 0
