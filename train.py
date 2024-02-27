import sys
sys.path.append(".") 

import os
import copy
import time

import numpy as np
import torch

from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import MTSAC
from torchrl.collector.async_mt_para_collector import AsyncMultiTaskParallelCollector
from torchrl.replay_buffers import SharedReplayBuffer
from torchrl.env import get_task_info

from torchrl.utils import get_parameters_num

def experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    params_copy = copy.deepcopy(params)
    params_copy['param_num'] = {}

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    params['general_setting']['device'] = device

    env_info = get_task_info(params["env"])
    params['general_setting']['env_info'] = env_info

    if args.cuda:
        torch.backends.cudnn.deterministic = True

    if 'base_type' in params['net']:
        params['net']['base_type']=networks.BASENET_DICT[params['net']['base_type']]
    
    # Build actor net
    pf = policies.POLICY_DICT[params['policy_net']](
        input_shape = env_info['obs_shape'], 
        output_shape = 2 * env_info['action_shape'][0],
        **params['net'])
    pf_param_num = get_parameters_num(pf.parameters())
    params_copy['param_num']['pf'] = pf_param_num

    # Build critic net
    qf_input_shape = env_info['obs_shape']
    qf_input_shape[0] += env_info['action_shape'][0]
    qf1 = networks.NETWORK_DICT[params['value_net']]( 
        input_shape = qf_input_shape,
        output_shape = 1,
        **params['net'])
    qf2 = networks.NETWORK_DICT[params['value_net']]( 
        input_shape = qf_input_shape,
        output_shape = 1,
        **params['net'])
    qf_param_num = get_parameters_num(qf1.parameters())
    params_copy['param_num']['qf'] = qf_param_num

    # Set logger
    experiment_name = os.path.split(os.path.splitext(args.config)[0])[-1] if args.id is None else args.id
    log_time = time.strftime("%Y-%m-%d,%H:%M:%S",time.gmtime())
    logger = Logger(experiment_name, params_copy['env']['env_name'], params_copy['algorithm'], args.seed, log_time, params_copy, args.log_dir)
    params['general_setting']['logger'] = logger
    eval_logger_args = dict(
        experiment_id = experiment_name, 
        env_name =  params_copy['env']['env_name'], 
        alg_name = params_copy['algorithm'], 
        seed = args.seed, 
        log_time = log_time,
        log_dir = args.log_dir,
    )

    # Build replay buffer
    example_ob = env_info['observation_sample']
    example_dict = { 
        "obs": (example_ob, np.float32),
        "next_obs": (example_ob, np.float32),
        "acts": (env_info['action_sample'], np.float32),
        "rewards": ([0], np.float32),
        "terminals": ([False], np.float32),
        "task_idxs": ([0], np.int32)
    }
    if params["algorithm"] in ["depthroute", "modelroute", "layerroute"]:
        if params["algorithm"] == "depthroute":
            gate_shape = params["net"]["module_num"] * (params["net"]["module_num"] + 1) // 2
        elif params["algorithm"] == "modelroute":
            gate_shape = params["net"]["model_num"]
        elif params["algorithm"] == "layerroute":
            each_layer_num = params["net"]["module_num"] // params["net"]["layer_num"]
            gate_shape = each_layer_num * ((params["net"]["layer_num"] - 1) * each_layer_num + 1)
        example_dict.update({
            "gates_onehot": (np.zeros(gate_shape), np.float32)
        })
    
    buffer_param = params['replay_buffer']
    replay_buffer = SharedReplayBuffer(
        int(buffer_param['size']),
        env_info['task_nums']
    )
    replay_buffer.build_by_example(example_dict)
    params['general_setting']['replay_buffer'] = replay_buffer

    params['general_setting']['save_dir'] = os.path.join(logger.work_dir,"model")
    params['general_setting']['num_epochs'] += 1    # TODO
    params['general_setting']['collector'] = AsyncMultiTaskParallelCollector(
        env_param = params['env'],
        env_seed = args.seed,
        task_names=env_info['task_names'],
        pf = pf,
        replay_buffer = replay_buffer,
        device = device,
        epoch_frames = params['general_setting']['epoch_frames'],
        max_episode_frames = params['general_setting']['max_episode_frames'],
        pre_train_epochs = params['general_setting']['pretrain_epochs'],
        train_epochs = params['general_setting']['num_epochs'],
        eval_epoch_interval = params['general_setting']['eval_epoch_interval'],
        eval_episodes = params['general_setting']['eval_episodes'],
        logger_args = eval_logger_args,
        save_dir = params['general_setting']['save_dir'],
    )

    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    agent = MTSAC(
        algo_name = params['algorithm'],
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()

if __name__ == "__main__":
    experiment(args)
