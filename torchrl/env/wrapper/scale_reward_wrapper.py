from .base_wrapper import BaseWrapper

class ScaleRewardWrapper(BaseWrapper):
    """
    Scale the reward
    """

    def __init__(self, env, reward_scale=1):
        super(ScaleRewardWrapper, self).__init__(env)
        self._reward_scale = reward_scale

    def reward(self, reward):
        if self.training:
            return self._reward_scale * reward
        else:
            return reward
        
    def step(self, act):
        obs, rew, done, truncated, info = self.env.step(act)
        return obs, self.reward(rew), done, truncated, info