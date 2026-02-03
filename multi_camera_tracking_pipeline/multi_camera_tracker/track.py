import numpy


class EKFTrack:
    next_track_id = 0

    def __init__(self, initial_measurement, time_step=1.0):
        # Set the unique ID
        self.id = EKFTrack.next_track_id
        EKFTrack.next_track_id += 1

        self.dt = time_step

        # Initialize state vector
        self.state = numpy.array([
            initial_measurement[0],  # px
            initial_measurement[1],  # py
            0.0,  # v
            0.0,  # theta
            0.0  # theta_dot
        ]).reshape(5, 1)

        # Initialize covariance matrix
        # We set a low uncertainty for position and high for velocity
        self.covariance = numpy.diag([1.0, 1.0, 1000.0, 1000.0, 1000.0])

        # Set counter for missed detections
        self.missed_frames = 0

    def predict(self, Q):
        """
        Predicts the next state of the track using the CTRV model
        :param Q: Process Noise Matrix (5x5)
        :return:
        """
        px, py, v, theta, theta_dot = self.state.flatten()

        # 1. State Prediction using f(x)
        if abs(theta_dot) > 0.001:  # Turning case
            # Calculate new state
            px_new = px + (v / theta_dot) * (numpy.sin(theta + theta_dot * self.dt) - numpy.sin(theta))
            py_new = py + (v / theta_dot) * (-numpy.cos(theta + theta_dot * self.dt) + numpy.cos(theta))
            v_new = v
            theta_new = theta + theta_dot * self.dt
            theta_dot_new = theta_dot
        else:
            # Calculate new state
            px_new = px + v * numpy.cos(theta) * self.dt
            py_new = py + v * numpy.sin(theta) * self.dt
            v_new = v
            theta_new = theta
            theta_dot_new = theta_dot

        self.state = numpy.array([px_new, py_new, v_new, theta_new, theta_dot_new]).reshape(5, 1)

        # 2. Jacobian Calculation
        F_j = numpy.zeros((5, 5))
        if abs(theta_dot) > 0.001:

            a = v / theta_dot
            b = theta + theta_dot * self.dt

            F_j[0, 0] = 1.0
            F_j[0, 1] = 0.0
            F_j[0, 2] = (1 / theta_dot) * (numpy.sin(b) - numpy.sin(theta))
            F_j[0, 3] = a * (numpy.cos(b) - numpy.cos(theta))
            F_j[0, 4] = self.dt * a * numpy.cos(b) - (a / theta_dot) * (
                        numpy.sin(b) - numpy.sin(theta))

            # d(py_new)/d(px, py, v, theta, theta_dot)
            F_j[1, 0] = 0.0
            F_j[1, 1] = 1.0
            F_j[1, 2] = (1 / theta_dot) * (-numpy.cos(b) + numpy.cos(theta))
            F_j[1, 3] = a * (numpy.sin(b) - numpy.sin(theta))
            F_j[1, 4] = self.dt * a * numpy.sin(b) - (a / theta_dot) * (
                        -numpy.cos(b) + numpy.cos(theta))

            # d(v_new)/...
            F_j[2, 2] = 1.0

            # d(theta_new)/...
            F_j[3, 3] = 1.0
            F_j[3, 4] = self.dt

            # d(theta_dot_new)/...
            F_j[4, 4] = 1.0

        else:  # Straight line case
            # --- Populate Jacobian ---
            # d(px_new)/d(px, py, v, theta, theta_dot)
            F_j[0, 0] = 1.0
            F_j[0, 2] = numpy.cos(theta) * self.dt
            F_j[0, 3] = -v * numpy.sin(theta) * self.dt

            # d(py_new)/...
            F_j[1, 1] = 1.0
            F_j[1, 2] = numpy.sin(theta) * self.dt
            F_j[1, 3] = v * numpy.cos(theta) * self.dt

            # d(v_new)/...
            F_j[2, 2] = 1.0

            # d(theta_new)/...
            F_j[3, 3] = 1.0

            # d(theta_dot_new)/...
            F_j[4, 4] = 1.0

            # --- 3. Predict the new covariance ---
            # P_pred = F_j * P * F_j.T + Q
        self.covariance = F_j @ self.covariance @ F_j.T + Q

    def update(self, measurement, H, R):
        """
        Update the track with a new measurement.
        :param measurement: The new measurement [x, y]
        :param H: Measurement Matrix
        :param R: Measurement Noise Matrix
        :return:
        """
        z = numpy.array(measurement).reshape(2, 1)

        # Compute the error: y = z - H * x_pred
        y = z - H @ self.state

        # Calculate the Kalman Gain:
        # K = P_pred * H.T * (H * P_pred * H.T + R)^-1

        # S = H * P_pred * H.T + R
        S = H @ self.covariance @ H.T + R

        # K = P_pred * H.T * S^-1
        K = self.covariance @ H.T @ numpy.linalg.inv(S)

        # Update the state: x_upd = x_pred + K * y
        self.state = self.state + K @ y

        # Update the covariance: P_upd = (I - K * H) * P_pred
        I = numpy.eye(self.covariance.shape[0])
        self.covariance = (I - K @ H) @ self.covariance
