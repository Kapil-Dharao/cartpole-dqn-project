"""CartPole-v1 physics, adapted from the public Gym/Gymnasium implementation.

This local environment keeps the project runnable without an installed Gymnasium.
When Gymnasium is installed, train.py can use the official CartPole-v1 instead.
"""
import math
import numpy as np


class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.steps = 0
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = map(float, self.state)
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # Gymnasium's default integrator is explicit Euler.
        self.state = np.array([x + self.tau*x_dot,
                               x_dot + self.tau*xacc,
                               theta + self.tau*theta_dot,
                               theta_dot + self.tau*thetaacc], dtype=np.float32)
        self.steps += 1
        terminated = bool(abs(self.state[0]) > self.x_threshold or
                          abs(self.state[2]) > self.theta_threshold_radians)
        truncated = self.steps >= 500
        return self.state.copy(), 1.0, terminated, truncated, {}

    def close(self):
        pass
