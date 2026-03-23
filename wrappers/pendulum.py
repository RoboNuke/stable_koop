import gym


class PendulumWrapper(gym.Wrapper):
    """Wrapper to override pendulum physical parameters (mass, length, gravity)."""

    def __init__(self, env, m=1.0, l=1.0, g=10.0):
        super().__init__(env)
        self.unwrapped.m = m
        self.unwrapped.l = l
        self.unwrapped.g = g


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    wrapped_env = PendulumWrapper(env, m=10, l=0.5, g=9.81)
    print(wrapped_env.m, wrapped_env.l, wrapped_env.g)