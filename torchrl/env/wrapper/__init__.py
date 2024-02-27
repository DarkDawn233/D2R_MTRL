from .base_wrapper import BaseWrapper
from .scale_reward_wrapper import ScaleRewardWrapper

wrapper_dict = {
    'base_wrapper': BaseWrapper,
    'scale_reward_wrapper': ScaleRewardWrapper,
}
