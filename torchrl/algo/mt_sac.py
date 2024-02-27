import numpy as np
import copy
import os
import time
from collections import deque

import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import torchrl.algo.utils as atu


class MTSAC():
    """
    Support Different Temperature for different tasks
    """
    def __init__(
            self,
            algo_name,
            env_info=None,
            replay_buffer = None,
            collector = None,
            logger = None,
            device = 'cpu',

            discount=0.99,
            num_epochs = 3000,
            epoch_frames = 1000,
            max_episode_frames = 999,
            eval_epoch_interval = 50,
            eval_episodes = 1,
            batch_size = 128,
            opt_times = 1,
            min_pool = 0,

            save_interval = 100,
            save_dir = None,
            pretrain_epochs=0,
            target_hard_update_period = 1000,
            use_soft_update = True,
            tau = 0.001,
            pf = None,
            qf1 = None,
            qf2 = None,
            taskf = None,
            plr = 1e-4,
            qlr = 1e-4,
            alphalr = 1e-4,
            tasklr = 1e-4,
            optimizer_class=optim.Adam,
            policy_std_reg_weight=0.,
            policy_mean_reg_weight=0.,
            reparameterization=True,
            automatic_entropy_tuning=True,
            max_log_alpha=None,
            min_log_alpha=None,
            target_entropy=None,
            temp_reweight=False,
            grad_clip=True,
            mask_extreme_task=False,
            task_loss_threshold=3e3,
            ):

        self.algo_name = algo_name
        self.env_info = env_info
        self.replay_buffer = replay_buffer
        self.collector = collector 
        self.logger = logger       
        self.device = device
        self.discount = discount
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.eval_epoch_interval = eval_epoch_interval

        self.batch_size = batch_size
        self.opt_times = opt_times
        self.min_pool = min_pool
        self.training_update_num = 0

        self.train_rewards_deque = deque(maxlen=32)
        self.train_success_deque = deque(maxlen=32)

        self.save_interval = save_interval
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.pretrain_epochs = pretrain_epochs
        
        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        self.tau = tau

        self.pf = pf
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()
        self.taskf = taskf

        self.to(self.device)

        self.optimizer_class = optimizer_class
        self.qf1_optimizer = optimizer_class(
            atu.get_optimize_params(self.qf1),
            lr=qlr,
        )
        self.qf2_optimizer = optimizer_class(
            atu.get_optimize_params(self.qf2),
            lr=qlr,
        )
        self.pf_optimizer = optimizer_class(
            atu.get_optimize_params(self.pf),
            lr=plr,
        )
        self.optimizers = [self.pf_optimizer, self.qf1_optimizer, self.qf2_optimizer]
        self.nets = [self.pf, self.qf1, self.qf2]

        if self.taskf is not None:
            self.task_optimizer = optimizer_class(
                atu.get_optimize_params(self.taskf),
                lr=tasklr,
            )
            self.optimizers.append(self.task_optimizer)
            self.nets.append(self.taskf)

        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight

        self.task_nums = env_info['task_nums']

        self.reparameterization = reparameterization
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.max_log_alpha = max_log_alpha
        self.min_log_alpha = min_log_alpha
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env_info['action_shape']).item()  # from rlkit
            self.log_alpha = torch.zeros(self.task_nums).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = self.optimizer_class(
                [self.log_alpha],
                lr=alphalr,
            )
            self.optimizers.append(self.alpha_optimizer)
            self.nets.append([self.log_alpha])
        
        self.sample_key = ["obs", "next_obs", "acts", "rewards",
                           "terminals",  "task_idxs"]
        if self.algo_name in ["depthroute", "modelroute", "layerroute"]:
            self.sample_key += ["gates_onehot"]

        self.temp_reweight = temp_reweight
        self.grad_clip = grad_clip

        # Paco Tricks
        self.mask_extreme_task = mask_extreme_task
        self.task_loss_threshold = task_loss_threshold

    def update(self, batch):
        self.training_update_num += 1
        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']
        task_idxs = batch['task_idxs']

        rewards = torch.tensor(rewards).to(self.device)
        terminals = torch.tensor(terminals).to(self.device)
        obs = torch.tensor(obs).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        next_obs = torch.tensor(next_obs).to(self.device)
        task_idxs = torch.tensor(task_idxs).to(self.device).squeeze(-1)

        gates_info = None
        if 'gates_onehot' in batch:
            gates_onehot = batch['gates_onehot']
            gates_onehot = torch.tensor(gates_onehot).to(self.device)
            gates_info = gates_onehot

        """
        Policy update.
        """
        self.pf.train()
        self.qf1.train()
        self.qf2.train()
        if self.taskf is not None:
            self.taskf.train()

        """
        Policy operations.
        """
        if self.algo_name in ["depthroute", "modelroute", "layerroute"]:
            sample_info = self.pf.explore(obs, idx=task_idxs, gate_sample=gates_info, return_log_probs=True, return_gate=True)
        else:
            sample_info = self.pf.explore(obs, idx=task_idxs, return_log_probs=True)

        mean = sample_info["mean"]
        log_std = sample_info["log_std"]
        new_actions = sample_info["action"]
        log_probs = sample_info["log_prob"]

        reweight_coeff = 1
        if self.automatic_entropy_tuning:
            """
            Alpha Loss
            """
            batch_size = log_probs.shape[0]
            log_alphas = (self.log_alpha.unsqueeze(0)).expand(
                            (batch_size, self.task_nums))
            log_alphas = log_alphas.unsqueeze(-1)

            if self.min_log_alpha is not None:
                log_alphas = torch.clamp_min(log_alphas, self.min_log_alpha)

            alpha_loss = -(log_alphas *
                           (log_probs + self.target_entropy).detach()).mean(0).squeeze(-1)


            clip_log_alpha = self.log_alpha.detach()
            if self.min_log_alpha is not None:
                clip_log_alpha = torch.clamp_min(clip_log_alpha, self.min_log_alpha)

            alphas = (clip_log_alpha.exp()).unsqueeze(0)
            alphas = alphas.expand((batch_size, self.task_nums)).unsqueeze(-1)
            
            if self.temp_reweight:
                softmax_temp = F.softmax(-self.log_alpha.detach()).unsqueeze(0)
                reweight_coeff = softmax_temp.expand((batch_size,
                                                      self.task_nums))
                reweight_coeff = reweight_coeff.unsqueeze(-1) * self.task_nums
        else:
            alphas = 1
            alpha_loss = 0

        """
        Policy Loss
        """
        copy_qf1 = copy.deepcopy(self.qf1)
        copy_qf2 = copy.deepcopy(self.qf2)

        if self.algo_name in ["depthroute", "modelroute", "layerroute"]:
            q_new_actions = torch.min(
                copy_qf1([obs, new_actions], explore=False, idx=task_idxs),
                copy_qf2([obs, new_actions], explore=False, idx=task_idxs))
        else:
            q_new_actions = torch.min(
                copy_qf1([obs, new_actions], idx=task_idxs),
                copy_qf2([obs, new_actions], idx=task_idxs))

        if not self.reparameterization:
            raise NotImplementedError
        else:
            assert log_probs.shape == q_new_actions.shape
            policy_loss = (reweight_coeff *
                           (alphas * log_probs - q_new_actions)).mean(0).squeeze(-1)

        std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean(0).mean(-1)
        mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean(0).mean(-1)

        policy_loss += std_reg_loss + mean_reg_loss

        """
        QF Loss
        """
        q1_pred = self.qf1([obs, actions], idx=task_idxs)
        q2_pred = self.qf2([obs, actions], idx=task_idxs)
        
        with torch.no_grad():
            if self.algo_name in ["depthroute", "modelroute", "layerroute"]:
                target_sample_info = self.pf.explore(next_obs, idx=task_idxs, gate_explore=False, return_log_probs=True)
            else:
                target_sample_info = self.pf.explore(next_obs, idx=task_idxs, return_log_probs=True)
            target_actions = target_sample_info["action"]
            target_log_probs = target_sample_info["log_prob"]

            if self.algo_name in ["depthroute", "modelroute", "layerroute"]:
                target_q1_pred = self.target_qf1([next_obs, target_actions], explore=False, idx=task_idxs)
                target_q2_pred = self.target_qf2([next_obs, target_actions], explore=False, idx=task_idxs)
            else:
                target_q1_pred = self.target_qf1([next_obs, target_actions], idx=task_idxs)
                target_q2_pred = self.target_qf2([next_obs, target_actions], idx=task_idxs)

            min_target_q = torch.min(target_q1_pred, target_q2_pred)
            target_v_values = min_target_q - alphas * target_log_probs

            # There is no actual terminate in meta-world -> just filter all time_limit terminal
            q_target = rewards + (1. - terminals) * self.discount * target_v_values

        qf1_loss = (reweight_coeff *
                    ((q1_pred - q_target.detach()) ** 2)).mean(0).squeeze(-1)
        qf2_loss = (reweight_coeff *
                    ((q2_pred - q_target.detach()) ** 2)).mean(0).squeeze(-1)              

        assert q1_pred.shape == q_target.shape, print(q1_pred.shape, q_target.shape)
        assert q2_pred.shape == q_target.shape, print(q1_pred.shape, q_target.shape)

        """
        Update Networks
        """

        task_loss = alpha_loss + policy_loss + qf1_loss + qf2_loss

        if self.mask_extreme_task:
            all_loss = self._mask_extreme_task(task_loss)
        else:
            all_loss = task_loss

        all_loss = all_loss.mean()

        for optimizer in self.optimizers:
            optimizer.zero_grad()
        all_loss.backward()
        norm_infos = []
        for optimizer, net in zip(self.optimizers, self.nets):
            norm_info = torch.nn.utils.clip_grad_norm_(atu.get_optimize_params(net), 1)
            norm_infos.append(norm_info)
            optimizer.step()
        
        # after update
        if self.max_log_alpha is not None:
            max_log_alpha_data = torch.min(self.log_alpha, torch.ones_like(self.log_alpha) * self.max_log_alpha)
            self.log_alpha.data.copy_(max_log_alpha_data.data)
        
        del copy_qf1
        del copy_qf2
        
        self._update_target_networks()
        if self.algo_name in ["depthroute", "modelroute", "layerroute"]:
            self._update_gumbel_temperature()

        # Information For Logger
        info = {}
        info['Training/Reward_Mean'] = rewards.mean().item()

        if self.automatic_entropy_tuning:
            for i in range(self.task_nums):
                info["Training/alpha_{}".format(i)] = self.log_alpha[i].exp().item()
            if hasattr(self.pf, "gumbel_temperature"):
                for i in range(self.task_nums):
                    info["Training/temp_{}".format(i)] = self.pf.gumbel_temperature[i].item()
            info["Training/alpha_loss"] = alpha_loss.mean().item()
        info['Training/policy_loss'] = policy_loss.mean().item()
        info['Training/qf1_loss'] = qf1_loss.mean().item()
        info['Training/qf2_loss'] = qf2_loss.mean().item()
        info['Training/loss'] = all_loss.item()

        if self.grad_clip:
            info['Training/pf_norm'] = norm_infos[0].item()
            info['Training/qf1_norm'] = norm_infos[1].item()
            info['Training/qf2_norm'] = norm_infos[2].item()
            if self.taskf is not None:
                info['Training/task_norm'] = norm_infos[3].item()

        return info
    
    def _update_gumbel_temperature(self):
        for net in self.networks:
            net.update_gumbel_temperature(self.log_alpha)

    def update_per_epoch(self):
        if self.replay_buffer.num_steps_can_sample() * self.task_nums > max( self.min_pool, self.batch_size ):
            for _ in range(self.opt_times):
                batch = self.replay_buffer.random_batch(self.batch_size,
                                                        self.sample_key,
                                                        reshape=False)
                infos = self.update(batch)
                self.logger.add_update_info(infos)
    
    def update_per_timestep(self):
        if self.replay_buffer.num_steps_can_sample() * self.task_nums > max( self.min_pool, self.batch_size ):
            for _ in range( self.opt_times ):
                batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
                infos = self.update( batch )
                self.logger.add_update_info( infos )
    
    def pretrain(self):

        total_frames = 0
        
        for pretrain_epoch in range(self.pretrain_epochs):

            start = time.time()
            
            training_epoch_info =  self.collector.train_one_epoch()
            self.train_rewards_deque.append(training_epoch_info["train_rewards"])
            self.train_success_deque.append(training_epoch_info["train_success"])

            total_frames += self.env_info['task_nums'] * self.epoch_frames
            
            infos = {}
            
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Train_Rewards"] = np.mean(self.train_rewards_deque)
            infos["Train_Success"] = np.mean(self.train_success_deque)
            
            self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos, csv_write=False)
        
        self.pretrain_frames = total_frames

        self.logger.log("Finished Pretrain")

    def snapshot(self, prefix, epoch):
        for name, network in self.snapshot_networks:
            model_file_name="model_{}_{}.pth".format(name, epoch)
            model_path=os.path.join(prefix, model_file_name)
            torch.save(network.state_dict(), model_path)

    def train(self):
        self.pretrain()
        self.total_frames = 0
        if hasattr(self, "pretrain_frames"):
            self.total_frames = self.pretrain_frames


        for epoch in range(self.num_epochs):

            if epoch % self.eval_epoch_interval == 0:
                # Send signal to eval
                self.collector.eval_one_epoch()
            
            start_time = time.time()

            # Collect Data
            training_epoch_info =  self.collector.train_one_epoch()
            self.train_rewards_deque.append(training_epoch_info["train_rewards"])
            self.train_success_deque.append(training_epoch_info["train_success"])
            explore_time = time.time() - start_time

            # Training
            train_start_time = time.time()
            self.update_per_epoch()
            train_time = time.time() - train_start_time

            # Info for logging
            infos = {}
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Train_Rewards"] = np.mean(self.train_rewards_deque)
            infos["Train_Success"] = np.mean(self.train_success_deque)

            infos["Explore_Time"] = explore_time
            infos["Train___Time"] = train_time

            self.total_frames += self.env_info['task_nums'] * self.epoch_frames

            self.logger.add_epoch_info(epoch, self.total_frames,
                time.time() - start_time, infos)

            if epoch != 0 and epoch % self.save_interval == 0:
                self.snapshot(self.save_dir, epoch)

        self.snapshot(self.save_dir, "finish")
        self.collector.terminate()
        self.logger.close()
    
    def _update_target_networks(self):
        if self.use_soft_update:
            for net, target_net in self.target_networks:
                atu.soft_update_from_to(net, target_net, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                for net, target_net in self.target_networks:
                    atu.copy_model_params_from_to(net, target_net)
        
    @property
    def networks(self):
        nets = [
            self.pf,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2
        ]
        if self.taskf is not None:
            nets.append(self.taskf)
        return nets
    
    @property
    def snapshot_networks(self):
        nets = [
            ["pf", self.pf],
            ["qf1", self.qf1],
            ["qf2", self.qf2],
        ]
        if self.taskf is not None:
            nets.append(["taskf", self.taskf])
        return nets

    @property
    def target_networks(self):
        return [
            ( self.qf1, self.target_qf1 ),
            ( self.qf2, self.target_qf2 )
        ]

    def to(self, device):
        for net in self.networks:
            net.to(device)
    
    def _mask_extreme_task(self, task_loss):
        extreme_task_mask = task_loss > self.task_loss_threshold
        extreme_task_mask = 1 - extreme_task_mask.float()
        return extreme_task_mask * task_loss
