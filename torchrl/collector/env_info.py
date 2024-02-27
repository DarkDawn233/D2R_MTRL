from torchrl.env import get_env

class EnvInfo():
    def __init__(self, 
            env_param,
            env_seed,
            task_names,
            device,
            epoch_frames,
            eval_epoch_interval,
            eval_episodes,
            max_episode_frames,
            ):

        self.current_step = 0

        self.env_param = env_param
        self.env_seed = env_seed
        self.task_names = task_names 
        self.device = device
        self.epoch_frames = epoch_frames
        self.eval_epoch_interval = eval_epoch_interval
        self.eval_episodes = eval_episodes
        self.max_episode_frames = max_episode_frames
    
    def make_env(self):
        self.env = get_env(self.env_param, seed=self.env_seed)
        self.task_nums = self.env.task_nums