from gym.vector.async_vector_env import AsyncVectorEnv, AsyncState
import multiprocessing as mp
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from copy import deepcopy

from gym.error import (
    NoAsyncCallError,
)
from gym.vector.utils import (
    concatenate,
)

class VecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_type,
        task_names,
        obs_mode,
        env_fns,
        shared_memory=False,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        assert env_type in ["metaworld"]
        self.env_type = env_type
        self.task_names = task_names
        self.obs_mode = obs_mode
        super().__init__(
            env_fns=env_fns,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.task_nums = len(env_fns)
    
    def _check_spaces(self):
        return
    
    def step_wait(
        self, timeout: Optional[Union[int, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out. If ``None``, the call to :meth:`step_wait` never times out.

        Returns:
             The batched environment step information, (obs, reward, terminated, truncated, info)

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`step_wait` was called without any prior call to :meth:`step_async`.
            TimeoutError: If :meth:`step_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `step_wait` has timed out after {timeout} second(s)."
            )

        observations_list, rewards, terminateds, truncateds, infos = [], [], [], [], {}
        successes = []
        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            obs, rew, terminated, truncated, info = result

            successes.append(success)
            observations_list.append(obs)
            rewards.append(rew)
            terminateds.append(terminated)
            truncateds.append(truncated)
            if "final_info" in info:
                final_info = info["final_info"]
                del info["final_info"]
                info.update(final_info)
            infos = self._add_info(infos, info, i)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space,
                observations_list,
                self.observations,
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )

    def get_obs_shape(self):
        if self.env_type == "metaworld":
            obs_shape = [39]
            if self.obs_mode == "vanilla":
                obs_shape.append(0)
            elif self.obs_mode == "onehot_id":
                obs_shape.append(self.task_nums)
            elif self.obs_mode == "roberta":
                task_encode_dim = 768
                obs_shape.append(task_encode_dim)
        else:
            raise NotImplementedError
        
        return obs_shape
