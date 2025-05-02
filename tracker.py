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
