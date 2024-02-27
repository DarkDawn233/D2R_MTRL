import numpy as np
import torch
import torch.multiprocessing as mp
import gym
import time
import os
import copy
from collections import OrderedDict
from .env_info import EnvInfo
from torchrl.utils import EvalLogger

class AsyncMultiTaskParallelCollector():

    def __init__(
            self,
            env_param,
            env_seed,
            task_names,
            pf,
            replay_buffer,
            device='cpu',
            epoch_frames=1000,
            max_episode_frames = 999,
            pre_train_epochs=0,
            train_epochs=1,
            eval_epoch_interval=32,
            eval_episodes=1,
            logger_args=None,
            save_dir=None):

        self.pf = pf
        self.replay_buffer = replay_buffer

        self.env_info = EnvInfo(
            env_param=env_param,
            env_seed=env_seed,
            task_names=task_names,
            device=device,
            epoch_frames=epoch_frames,
            eval_epoch_interval=eval_epoch_interval,
            eval_episodes=eval_episodes,
            max_episode_frames=max_episode_frames,
        )

        self.shared_funcs = copy.deepcopy(self.funcs)
        for key in self.shared_funcs:
            self.shared_funcs[key].to(self.env_info.device)

        self.manager = mp.Manager()

        self.pre_train_epochs = pre_train_epochs
        self.train_epochs = train_epochs
        self.logger_args = logger_args
        self.save_dir = save_dir

        self.start_worker()

    @classmethod
    def take_actions(
            cls, 
            funcs, 
            env_info, 
            ob_info,
            test=False,
            replay_buffer=None):

        pf = funcs["pf"]
        ob = ob_info["ob"]
        task_idx = ob_info["task_idx"]

        pf.eval()

        ob_tensor = torch.tensor(ob, dtype=torch.float32).to(env_info.device)
        task_idx_tensor = torch.tensor(task_idx, dtype=torch.int32).to(env_info.device)

        with torch.no_grad():
            if test:
                out = pf.eval_act(ob_tensor, task_idx_tensor)
            else:
                out = pf.explore(ob_tensor, task_idx_tensor)

        act = out["action"]
        act = act.detach().cpu().numpy()
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, _, info = env_info.env.step(act)
        env_info.current_step += 1

        if "gates_onehot" in out:
            gates = out["gates"].detach().cpu().numpy()
            gates_onehot = out["gates_onehot"].detach().cpu().numpy()
            info["gates"] = gates
            info["gates_onehot"] = gates_onehot

        if not test:
            sample_dict = {
                "obs": ob,
                "next_obs": next_ob,
                "acts": act,
                "task_idxs": task_idx.reshape(env_info.task_nums, 1),
                "rewards": reward.reshape(env_info.task_nums, 1),
                "terminals": done.reshape(env_info.task_nums, 1)
            }
            if "gates" in out:
                sample_dict.update({
                    "gates_onehot": gates_onehot,
                })
            replay_buffer.add_sample(sample_dict)
            
        return next_ob, done, reward, info

    @staticmethod
    def train_worker_process(
            cls,
            shared_funcs, 
            env_info,
            replay_buffer, 
            shared_que,
            epochs,
            pre_epochs
            ):

        replay_buffer.rebuild_from_tag()

        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)

        env_info.make_env()
        obs, info = env_info.env.reset()
        c_ob = {
            "ob": obs,
            "task_idx": np.array(info['task_idxs']),
        }

        task_nums = env_info.env.task_nums

        train_rew = np.zeros(task_nums)
        success = np.zeros(task_nums)

        current_epoch, current_pre_epoch = 0, 0

        while True:

            for key in shared_funcs:
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            train_rews, successes = [], []
            train_epoch_reward = np.zeros(task_nums)

            for _ in range(env_info.epoch_frames):
                next_ob, done, reward, info = cls.take_actions(local_funcs, env_info, c_ob, test=False, replay_buffer=replay_buffer )
                c_ob["ob"] = next_ob
                train_rew += reward
                train_epoch_reward += reward
                success = np.maximum(success, info["success"])
                for i in range(len(done)):
                    if done[i]:
                        train_rews.append(train_rew[i])
                        train_rew[i] = 0
                        successes.append(success[i])
                        success[i] = 0

            shared_que.put({
                'train_rewards': np.mean(train_rews),
                'train_epoch_reward': np.mean(train_epoch_reward),
                'train_success': np.mean(successes),
            })

            if current_pre_epoch < pre_epochs:
                current_pre_epoch += 1
            else:
                current_epoch += 1

            if current_epoch > epochs:
                break
        
        env_info.env.close_extras(terminate=True)

    
    @staticmethod
    def eval_worker_process(
            cls,
            shared_funcs, 
            env_info,
            eval_start_barrier, 
            epochs,
            pre_epochs,
            logger_args,
            save_dir
            ):

        logger = EvalLogger(**logger_args)

        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)

        env_info.make_env()
        obs, info = env_info.env.reset()
        c_ob = {
            "ob": obs,
            "task_idx": np.array(info['task_idxs']),
        }

        task_nums = env_info.env.task_nums

        current_epoch = 0
        total_frames = pre_epochs * env_info.task_nums * env_info.epoch_frames

        best_rewards = None
        best_success = None 

        while True:
            
            eval_start_barrier.wait()

            start_time = time.time()

            for key in shared_funcs:
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            eval_rews = np.zeros(task_nums) 
            eval_success = np.zeros(task_nums)

            info_gates = None
            info_gates_onehot = None
            info_gates_num = 0
            for _ in range(env_info.eval_episodes):
                episode_rew = np.zeros(task_nums)
                done = np.zeros(task_nums, dtype='bool')
                episode_success = np.zeros(task_nums)

                obs, info = env_info.env.reset()

                c_ob["ob"] = obs

                while not all(done):
                    next_ob, src_done, reward, info = cls.take_actions(local_funcs, env_info, c_ob, test=True)
                    c_ob["ob"] = next_ob
                    episode_rew += reward * (1-done)
                    success_src = info["success"] * (1-done)
                    episode_success = np.maximum(episode_success, success_src)
                    if "gates" in info:
                        if info_gates is None:
                            info_gates = info["gates"] * np.expand_dims(1-done, -1)
                            info_gates_onehot = info["gates_onehot"] * np.expand_dims(1-done, -1)
                            info_gates_num += 1
                        else:
                            info_gates += info["gates"] * np.expand_dims(1-done, -1)
                            info_gates_onehot += info["gates_onehot"] * np.expand_dims(1-done, -1)
                            info_gates_num += 1

                    done = np.logical_or(done, src_done)
                    
                eval_rews += episode_rew
                eval_success += episode_success

            eval_info = {
                'eval_rewards': eval_rews / env_info.eval_episodes,
                'eval_success': eval_success / env_info.eval_episodes,
                'task_names': env_info.env.task_names
            }

            # Eval log
            cls.log_eval_info(logger, eval_info, current_epoch, total_frames, time.time() - start_time)
            # if info_gates is not None:
            #     logger.save_gates_info("gates", info_gates / info_gates_num)
            #     logger.save_gates_info("gates_onehot", info_gates_onehot / info_gates_num)

            eval_rewards = np.mean(eval_info['eval_rewards'])
            eval_success = np.mean(eval_info['eval_success'])
            if best_rewards is None or \
                eval_success > best_success or \
                eval_success == best_success and eval_rewards > best_rewards:

                best_success = eval_success
                best_rewards = eval_rewards
                cls.snapshot(local_funcs, save_dir, 'best')

            current_epoch += env_info.eval_epoch_interval
            total_frames += env_info.eval_epoch_interval * env_info.task_nums * env_info.epoch_frames
            
            if current_epoch > epochs:
                break
        
        logger.close()
        env_info.env.close_extras(terminate=True)

        
    @classmethod
    def log_eval_info(cls, logger, eval_info, epoch, total_frames, time_consumed):

        task_names = eval_info["task_names"]
        eval_rews = eval_info["eval_rewards"]
        eval_successes = eval_info["eval_success"]

        dic = {}
        eval_rewards = np.mean(eval_rews)
        eval_success = np.mean(eval_successes)
        dic['Eval_Rewards'] = eval_rewards
        dic['Eval_Success'] = eval_success

        for task_name, eval_success, eval_rewards in zip(task_names, eval_successes, eval_rews):
            dic["Task/"+task_name+"_success"] = eval_success
            dic["Task/"+task_name+"_rewards"] = eval_rewards

        logger.add_epoch_eval_info(epoch, total_frames, time_consumed, dic)

    @classmethod
    def snapshot(cls, funcs, prefix, epoch):
        for name, network in funcs.items():
            model_file_name="model_{}_{}.pth".format(name, epoch)
            model_path=os.path.join(prefix, model_file_name)
            torch.save(network.state_dict(), model_path)

    def start_worker(self):

        self.shared_que = self.manager.Queue(1)
        env_info = copy.deepcopy(self.env_info)
        self.worker = mp.Process(
            target=self.__class__.train_worker_process,
            args=(
                self.__class__,
                self.shared_funcs,
                env_info,
                self.replay_buffer, 
                self.shared_que,
                self.train_epochs,
                self.pre_train_epochs
            )
        )
        self.worker.start()

        self.eval_start_barrier = mp.Barrier(2)
        env_info = copy.deepcopy(self.env_info)
        self.eval_worker = mp.Process(
            target=self.__class__.eval_worker_process,
            args=(
                self.__class__,
                self.shared_funcs,
                env_info,
                self.eval_start_barrier,
                self.train_epochs,
                self.pre_train_epochs,
                self.logger_args,
                self.save_dir
            )
        )
        self.eval_worker.start()
    
    def terminate(self):
        self.worker.join()
        self.eval_worker.join()

    def eval_one_epoch(self):

        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())

        self.eval_start_barrier.wait()

    def train_one_epoch(self):

        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
        worker_rst = self.shared_que.get()

        train_rews = worker_rst["train_rewards"]
        train_epoch_reward = worker_rst["train_epoch_reward"]
        train_successes = worker_rst["train_success"]

        return {
            'train_rewards': train_rews,
            'train_epoch_reward': train_epoch_reward,
            'train_success': train_successes,
        }
    
    def to(self, device):
        for func in self.funcs:
            self.funcs[func].to(device)
    
    @property
    def funcs(self):
        return {
            "pf": self.pf
        }
    
    @property
    def snapshot_networks(self):
        nets = [
            ["pf", self.pf],
        ]
        return nets

