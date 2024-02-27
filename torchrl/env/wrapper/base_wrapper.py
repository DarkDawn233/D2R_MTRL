import gym

class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseWrapper, self).__init__(env)
        self._wrapped_env = env
        self.training = True

    def train(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.train()
        self.training = True

    def eval(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.eval()
        self.training = False

    def render(self, mode='human', **kwargs):
        return self._wrapped_env.render(mode=mode, **kwargs)

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)