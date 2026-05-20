"""Run the exact diagnostic snippet the user requested."""

import gymnasium as gym
import numpy as np
from scipy.signal import place_poles

from policy.inverted_pendulum.lqr import _continuous_dynamics, _zoh_discretize, _DT


env = gym.make("InvertedPendulum-v5")
A_c, B_c = _continuous_dynamics()
A_d, B_d = _zoh_discretize(A_c, B_c, _DT)

print(f"dt = {env.unwrapped.dt}")
print(f"A_c =\n{A_c}")
print(f"B_c =\n{B_c}")
print(f"A_d =\n{A_d}")
print(f"B_d =\n{B_d}")
print(f"Open-loop eigenvalues of A_d: {np.linalg.eigvals(A_d)}")
# Sanity check: place poles manually at conservative locations
result = place_poles(A_d, B_d, [0.95, 0.9, 0.85, 0.8])
print(f"Pole-placement F: {result.gain_matrix}")
print(f"Max |u| at theta=0.01: {np.abs(result.gain_matrix @ np.array([0, 0, 0.01, 0]))}")

env.close()
