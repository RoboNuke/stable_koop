import gymnasium as gym
import numpy as np


class PendulumWrapper(gym.Wrapper):
    """Wrapper to override pendulum physical parameters (mass, length, gravity)."""

    def __init__(self, env, m=1.0, l=1.0, g=10.0):
        super().__init__(env)
        self.unwrapped.m = m
        self.unwrapped.l = l
        self.unwrapped.g = g


class FrictionPendulumWrapper(gym.Wrapper):
    """Adds viscous friction to the pendulum dynamics.

    Modifies the equation of motion to include a friction term:
        θ̈ = 3g/(2l) * sin(θ) + 3/(ml²) * u - B * θ̇

    The friction is injected into the physics step directly, not applied
    as a post-hoc correction.

    Args:
        env: gym pendulum environment
        friction_coeff: B, the friction coefficient
    """

    def __init__(self, env, friction_coeff=0.1):
        super().__init__(env)
        self.friction_coeff = friction_coeff

    def step(self, action):
        penv = self.unwrapped
        th, thdot = penv.state

        g = penv.g
        m = penv.m
        l = penv.l
        dt = penv.dt

        u = np.clip(action, -penv.max_torque, penv.max_torque)[0]
        penv.last_u = u

        # Original acceleration + friction term
        accel = 3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u
        accel -= self.friction_coeff * thdot

        newthdot = thdot + accel * dt
        newthdot = np.clip(newthdot, -penv.max_speed, penv.max_speed)
        newth = th + newthdot * dt

        penv.state = np.array([newth, newthdot])

        # Reuse gymnasium's cost function
        from gymnasium.envs.classic_control.pendulum import angle_normalize
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        if penv.render_mode == "human":
            penv.render()

        return penv._get_obs(), -costs, False, False, {}


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    wrapped_env = PendulumWrapper(env, m=10, l=0.5, g=9.81)
    print(wrapped_env.m, wrapped_env.l, wrapped_env.g)