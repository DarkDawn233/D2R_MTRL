from gym import Env
from gym.spaces import Box
import numpy as np
from pathlib import Path
import json5 as json

class MultiTaskEnv(Env):
    def __init__(self,
                 env_type,
                 task_envs,
                 task_names,
                 init_task_id=0,
                 sample_strategy="random",
                 obs_mode="vanilla",
                 max_episode_steps=200):
        assert env_type in ["metaworld"]
        assert sample_strategy in ["fixed", "random"]
        assert obs_mode in ["vanilla", "onehot_id", "roberta"]
        self.env_type = env_type
        self.task_envs = task_envs
        self.task_names = task_names
        self.task_nums = len(task_names)
        self.active_env_id = init_task_id
        self.sample_strategy = sample_strategy
        self.obs_mode = obs_mode
        self.task_encode = None
        if obs_mode == "roberta":
            assert env_type == "metaworld"
            roberta_json_path = Path(__file__).parent / "metaworld_env" / "roberta_encode.json"
            with open(roberta_json_path, "rb") as f:
                self.task_encode = json.load(f)

        self.max_episode_steps = max_episode_steps
        self.cur_step = 0

        self.train_mode = True
    
    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
    
    def get_task_idx(self):
        return self.active_env_id
    
    def get_task_nums(self):
        return self.task_nums
    
    def get_obs_shape(self):
        obs_shape = [self.task_envs[self.active_env_id].observation_space.shape[0]]
        if self.obs_mode == "vanilla":
            obs_shape.append(0)
        elif self.obs_mode == "onehot_id":
            obs_shape.append(self.task_nums)
        elif self.obs_mode == "roberta":
            task_encode_len = len(self.task_encode[self.task_names[self.active_env_id]])
            obs_shape.append(task_encode_len)
        return obs_shape

    def __active_task_onehot(self):
        one_hot = np.zeros(self.task_nums, dtype='float32')
        one_hot[self.active_env_id] = 1
        return one_hot

    def __augment_observation(self, obs):
        if self.obs_mode == "vanilla":
            pass
        elif self.obs_mode == "onehot_id":
            obs = np.concatenate([obs, self.__active_task_onehot()])
        elif self.obs_mode == "roberta":
            task_encode_vec = np.array(self.task_encode[self.task_names[self.active_env_id]], dtype='float32')
            obs = np.concatenate([obs, task_encode_vec])
        else:
            raise NotImplementedError
        return obs
    
    def __augment_information(self, info):
        info['task_idxs'] = self.active_env_id
        info['task_name'] = self.task_names[self.active_env_id]
        return info
            
    def reset(self):
        self.cur_step = 0
        if self.sample_strategy == "fixed":
            pass
        elif self.sample_strategy == "random":
            self.active_env_id = np.random.randint(0, self.task_nums)
        else:
            raise NotImplementedError
        obs = self.task_envs[self.active_env_id].reset()
        return self.__augment_observation(obs)
    
    def reset_with_index(self, task_id):
        self.cur_step = 0
        assert task_id >= 0 and task_id < self.task_nums
        self.active_env_id = task_id
        obs = self.task_envs[self.active_env_id].reset()
        return self.__augment_observation(obs)
    
    def step(self, action):
        obs, reward, done, info = self.task_envs[self.active_env_id].step(action)
        obs = self.__augment_observation(obs)
        info = self.__augment_information(info)
        self.cur_step += 1
        if self.cur_step >= self.max_episode_steps:
            done = True
        return obs, reward, done, info
    
    def render(self, mode="human"):
        return self.task_envs[self.active_env_id].render(mode=mode)
    
    @property
    def observation_space(self):
        if self.obs_mode == "vanilla":
            return self.task_envs[self.active_env_id].observation_space
        elif self.obs_mode == "onehot_id":
            src_high = self.task_envs[self.active_env_id].observation_space.high
            src_low = self.task_envs[self.active_env_id].observation_space.low
            onehot_high = np.ones(shape=(self.task_nums,))
            onehot_low = np.zeros(shape=(self.task_nums,))
            return Box(
                high=np.concatenate([src_high, onehot_high,]),
                low=np.concatenate([src_low, onehot_low,]))
        else:
            raise NotImplementedError

    @property
    def action_space(self):
        return self.task_envs[self.active_env_id].action_space

        
        

                 