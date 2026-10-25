"""Landau-de Gennes model with five components and the discrete L2 metric."""

import numpy as np


class LdGModel:
    """Landau-de Gennes model with five components and the discrete L2 metric."""

    def __init__(self, lambda_val: float = 15.0, h: float = 1 / 50):
        self.lam = lambda_val
        self.h = h
        self.h2 = h**2
        grid_points = int(2.0 / h) + 1
        self.I = grid_points - 2
        self.N = self.I**2
        self.state_size = 5 * self.N
        self.B = 6400.0
        self.C = 3500.0
        self.A = -self.B**2 / (3 * self.C)
        self.s = (self.B + np.sqrt(self.B**2 + 24 * abs(self.A) * self.C)) / (
            4 * self.C
        )
        self.A2C = self.A / (2 * self.C)
        self.A4C = self.A / (4 * self.C)
        self.B2C = self.B / (2 * self.C)
        self.M_weights = (
            np.concatenate(
                [
                    np.full(self.N, 2.0),
                    np.full(self.N, 2.0),
                    np.full(self.N, 6.0),
                    np.full(self.N, 2.0),
                    np.full(self.N, 2.0),
                ]
            )
            * self.h2
        )
        self.E0 = (
            self.s**2 * self.A / (6.0 * self.C)
            - self.s**3 * self.B / (27.0 * self.C)
            + self.s**4 / 18.0
        )

    def energy(self, u: np.ndarray) -> float:
        I = self.I
        N = self.N
        s = self.s
        q1 = u[0:N].reshape((I, I), order="F")
        q2 = u[N : 2 * N].reshape((I, I), order="F")
        q3 = u[2 * N : 3 * N].reshape((I, I), order="F")
        q4 = u[3 * N : 4 * N].reshape((I, I), order="F")
        q5 = u[4 * N : 5 * N].reshape((I, I), order="F")

        def horizontal_energy(q, left, right):
            A = np.hstack([np.full((I, 1), left), q])
            B = np.hstack([q, np.full((I, 1), right)])
            return np.sum((A - B) ** 2)

        def vertical_energy(q, top, bottom):
            A = np.vstack([np.full((1, I), top), q])
            B = np.vstack([q, np.full((1, I), bottom)])
            return np.sum((A - B) ** 2)

        Ee = (
            horizontal_energy(q1, s / 2, s / 2)
            + horizontal_energy(q2, 0.0, 0.0)
            + 3.0 * horizontal_energy(q3, -s / 6, -s / 6)
            + horizontal_energy(q4, 0.0, 0.0)
            + horizontal_energy(q5, 0.0, 0.0)
            + vertical_energy(q1, -s / 2, -s / 2)
            + vertical_energy(q2, 0.0, 0.0)
            + 3.0 * vertical_energy(q3, -s / 6, -s / 6)
            + vertical_energy(q4, 0.0, 0.0)
            + vertical_energy(q5, 0.0, 0.0)
        )
        trQ2 = 2.0 * q1**2 + 2.0 * q2**2 + 6.0 * q3**2 + 2.0 * q4**2 + 2.0 * q5**2
        cubic_part = (
            q1 * (q4**2 - q5**2)
            + 2.0 * q2 * q4 * q5
            + q3 * (-2.0 * q1**2 - 2.0 * q2**2 + 2.0 * q3**2 + q4**2 + q5**2)
        )
        Eb = self.A4C * trQ2 - self.B2C * cubic_part + trQ2**2 / 8.0 - self.E0
        return float(Ee + self.lam * np.sum(Eb) * self.h2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        I, N = (self.I, self.N)
        q1 = x[0:N].reshape((I, I), order="F")
        q2 = x[N : 2 * N].reshape((I, I), order="F")
        q3 = x[2 * N : 3 * N].reshape((I, I), order="F")
        q4 = x[3 * N : 4 * N].reshape((I, I), order="F")
        q5 = x[4 * N : 5 * N].reshape((I, I), order="F")
        q12q22 = q1**2 + q2**2
        q32 = q3**2
        q42q52 = q4**2 + q5**2
        trQ2 = q12q22 + 3 * q32 + q42q52

        def laplacian(q_mat, top, bottom, left, right):
            q_up = np.vstack([np.full((1, I), top), q_mat[:-1, :]])
            q_down = np.vstack([q_mat[1:, :], np.full((1, I), bottom)])
            q_left = np.hstack([np.full((I, 1), left), q_mat[:, :-1]])
            q_right = np.hstack([q_mat[:, 1:], np.full((I, 1), right)])
            return -(q_up + q_down + q_left + q_right - 4 * q_mat) / self.h2

        LB1 = laplacian(
            q1, -self.s / 2, -self.s / 2, self.s / 2, self.s / 2
        ) + self.lam * (
            self.A2C * q1 + self.B2C * (2 * q1 * q3 - 0.5 * (q4**2 - q5**2)) + trQ2 * q1
        )
        LB2 = laplacian(q2, 0, 0, 0, 0) + self.lam * (
            self.A2C * q2 + self.B2C * (2 * q2 * q3 - q4 * q5) + trQ2 * q2
        )
        LB3 = laplacian(
            q3, -self.s / 6, -self.s / 6, -self.s / 6, -self.s / 6
        ) + self.lam * (
            self.A2C * q3 + self.B2C * (q12q22 / 3.0 - q32 - q42q52 / 6.0) + trQ2 * q3
        )
        LB4 = laplacian(q4, 0, 0, 0, 0) + self.lam * (
            self.A2C * q4 - self.B2C * (q3 * q4 + q1 * q4 + q2 * q5) + trQ2 * q4
        )
        LB5 = laplacian(q5, 0, 0, 0, 0) + self.lam * (
            self.A2C * q5 - self.B2C * (q3 * q5 - q1 * q5 + q2 * q4) + trQ2 * q5
        )
        return np.concatenate(
            [
                LB1.flatten(order="F"),
                LB2.flatten(order="F"),
                LB3.flatten(order="F"),
                LB4.flatten(order="F"),
                LB5.flatten(order="F"),
            ]
        )

    def hvp(self, x: np.ndarray, v: np.ndarray, eps: float = 1e-06) -> np.ndarray:
        return (self.gradient(x + eps * v) - self.gradient(x - eps * v)) / (2 * eps)

    def norm(self, x: np.ndarray) -> float:
        return np.sqrt(np.dot(x, self.M_weights * x))
