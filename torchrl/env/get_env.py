from .metaworld_env import get_metaworld_envs, MetaworldVecEnv
from .multitask_env import MultiTaskEnv
from .wrapper import wrapper_dict
from .vec_env import VecEnv

def _make_env_func_list(task_envs, task_names, task_nums, obs_mode, wrapper_param):

    def _make_env_func(env_cls, args, wrapper_param=None):
        # global _make_env
        def _make_env():
            env = env_cls(**args)
            if wrapper_param is not None:
                for wrapper in wrapper_param:
                    wrapper_name = wrapper["name"]
                    if wrapper_name not in wrapper_dict:
                        raise ValueError(f"No wrapper named {wrapper_name}")
                    env = wrapper_dict[wrapper_name](env, **wrapper["kwargs"])
            return env
        return _make_env
    
    funcs = []
    for task_id, (task_env, task_name) in enumerate(zip(task_envs, task_names)):
        args = dict(
            env=task_env,
            task_name=task_name,
            task_id=task_id,
            task_nums=task_nums,
            obs_mode=obs_mode,
        )
        funcs.append(_make_env_func(MetaworldVecEnv, args, wrapper_param))
    
    return funcs


def get_env(env_param, seed=0):
    env_type = env_param["env_type"]
    assert env_type in ["metaworld"], "Error Config: 'env/env_type' missmatching"
    env_name = env_param["env_name"]
    obs_mode = env_param["obs_mode"]
    if env_type == "metaworld":
        if env_name == "mt1":
            assert "single_task_name" in env_param, "Error Config: 'mt1' requires 'single_task_name' param"
        random_goal = env_param["random_goal"]
        single_task_name = env_param["single_task_name"] if env_name == "mt1" else None
        
        task_envs, task_names = get_metaworld_envs(env_name, random_goal=random_goal, single_task_name=single_task_name, seed=seed)
        task_nums = len(task_envs)
        wrapper_param = env_param.get("wrapper", None)

    funcs = _make_env_func_list(task_envs, task_names, task_nums, obs_mode, wrapper_param)
        
    return VecEnv(env_type=env_type, task_names=task_names, obs_mode=obs_mode, env_fns=funcs, context="spawn")


def get_task_info(env_param):
    env_type = env_param["env_type"]
    assert env_type in ["metaworld"], "Error Config: 'env/env_type' missmatching"
    env_name = env_param["env_name"]
    obs_mode = env_param["obs_mode"]

    if env_type == "metaworld":
        if env_name == "mt1":
            assert "single_task_name" in env_param, "Error Config: 'mt1' requires 'single_task_name' param"
        random_goal = env_param["random_goal"]
        single_task_name = env_param["single_task_name"] if env_name == "mt1" else None
        
        task_envs, task_names = get_metaworld_envs(env_name, random_goal=random_goal, single_task_name=single_task_name)
        task_nums = len(task_envs)
        wrapper_param = env_param.get("wrapper", None)

        example_env = MetaworldVecEnv(env=task_envs[0], task_name=task_names[0], task_id=0, task_nums=task_nums, obs_mode=obs_mode)
        obs_shape = example_env.get_obs_shape()
        action_shape = example_env.action_space.shape
        observation_sample = example_env.observation_space.sample()
        action_sample = example_env.action_space.sample()
        del example_env
    
    task_info = dict(
        task_nums=task_nums,
        task_names=task_names,
        obs_shape=obs_shape,
        action_shape=action_shape,
        observation_sample=observation_sample,
        action_sample=action_sample,
    )

    return task_info

    



