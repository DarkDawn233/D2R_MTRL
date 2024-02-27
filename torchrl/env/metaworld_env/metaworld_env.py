import metaworld
import random
import numpy as np

from gym import Env
from gym.spaces import Box
from pathlib import Path
import json5 as json

def get_metaworld_envs(env_name,
                       random_goal=False,
                       single_task_name="pick-place-v2",
                       seed=0):

    if env_name == "mt10":
        mt_benchmark = metaworld.MT10(seed=seed)
    elif env_name == "mt50":
        mt_benchmark = metaworld.MT50(seed=seed)
    elif env_name == "mt1":
        mt_benchmark = metaworld.MT1(single_task_name, seed=seed)
    elif env_name == "mt2-test":
        mt_benchmark = metaworld.MT10(seed=seed)
    else:
        raise ValueError(f"No Metaworld Environment Name {env_name}")
    task_names, task_envs = [], []
    random.seed(seed)

    cnt = 0
    for task_name, task_cls in mt_benchmark.train_classes.items():
        env = task_cls()
        task = random.choice([task for task in mt_benchmark.train_tasks
                        if task.env_name == task_name])
        env.set_task(task)
        env._freeze_rand_vec = not random_goal
        task_envs.append(env)
        task_names.append(task_name)
        
        cnt += 1
        if env_name == "mt2-test" and cnt == 2:
            break
    
    return task_envs, task_names


class MetaworldVecEnv(Env):
    def __init__(self,
                 env,
                 task_name,
                 task_id=0,
                 task_nums=1,
                 obs_mode="vanilla",
                 max_episode_steps=200,
                 ):
        assert obs_mode in ["vanilla", "onehot_id", "roberta"]
        self.env = env
        self.task_name = task_name
        self.task_id = task_id
        self.task_nums = task_nums
        self.obs_mode = obs_mode
        self.task_encode = None
        if obs_mode == "roberta":
            roberta_json_path = Path(__file__).parent / "roberta_encode.json"
            with open(roberta_json_path, "rb") as f:
                self.task_encode = json.load(f)[self.task_name]

        self.max_episode_steps = max_episode_steps
        self.cur_step = 0
    
    def get_task_idx(self):
        return self.task_id

    def get_obs_shape(self):
        obs_shape = [39]
        if self.obs_mode == "vanilla":
            obs_shape.append(0)
        elif self.obs_mode == "onehot_id":
            obs_shape.append(self.task_nums)
        elif self.obs_mode == "roberta":
            task_encode_dim = 768
            obs_shape.append(task_encode_dim)

        return obs_shape

    def __active_task_onehot(self):
        one_hot = np.zeros(self.task_nums, dtype='float32')
        one_hot[self.task_id] = 1
        return one_hot

    def __augment_observation(self, obs):
        if self.obs_mode == "vanilla":
            pass
        elif self.obs_mode == "onehot_id":
            obs = np.concatenate([obs, self.__active_task_onehot()])
        elif self.obs_mode == "roberta":
            task_encode_vec = np.array(self.task_encode, dtype='float32')
            obs = np.concatenate([obs, task_encode_vec])
        else:
            raise NotImplementedError
        return obs
    
    def __augment_information(self, info):
        info['task_idxs'] = self.task_id
        info['task_name'] = self.task_name
        return info
            
    def reset(self):
        self.cur_step = 0
        obs = self.env.reset()
        info = self.__augment_information({})
        return self.__augment_observation(obs), info 
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.__augment_observation(obs)
        info = self.__augment_information(info)
        self.cur_step += 1
        if self.cur_step >= self.max_episode_steps:
            done = True
        return obs, reward, done, False, info
    
    def render(self, mode="human"):
        return self.task_envs[self.active_env_id].render(mode=mode)
    
    @property
    def observation_space(self):
        if self.obs_mode == "vanilla":
            return self.env.observation_space
        elif self.obs_mode == "onehot_id":
            src_high = self.env.observation_space.high
            src_low = self.env.observation_space.low
            onehot_high = np.ones(shape=(self.task_nums,))
            onehot_low = np.zeros(shape=(self.task_nums,))
            return Box(
                high=np.concatenate([src_high, onehot_high,]),
                low=np.concatenate([src_low, onehot_low,]))
        elif self.obs_mode == "roberta":
            src_high = self.env.observation_space.high
            src_low = self.env.observation_space.low
            encode_high = np.ones_like(np.array(self.task_encode, dtype=np.float32))
            encode_low = encode_high * -1
            return Box(
                high=np.concatenate([src_high, encode_high,]),
                low=np.concatenate([src_low, encode_low,]))
        else:
            raise NotImplementedError

    @property
    def action_space(self):
        return self.env.action_space
    