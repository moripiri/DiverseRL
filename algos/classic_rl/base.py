from gymnasium import spaces


class ClassicRL:
    def __init__(self, env):
        assert isinstance(env.observation_space, (spaces.Discrete, spaces.Tuple))
        if isinstance(env.observation_space, spaces.Tuple):
            for item in env.observation_space:
                assert isinstance(item, spaces.Discrete)

        assert isinstance(env.action_space, spaces.Discrete)

        self.state_dim = env.observation_space.n if isinstance(env.observation_space, spaces.Discrete) \
            else tuple(map(lambda x: x.n, env.observation_space))

        self.action_dim = env.action_space.n

    def __repr__(self):
        return "ClassicAlgorithm"

    def get_action(self, observation):
        raise NotImplementedError

    def train(self, observation, action, reward, next_observation, terminated, truncated, info):
        raise NotImplementedError
