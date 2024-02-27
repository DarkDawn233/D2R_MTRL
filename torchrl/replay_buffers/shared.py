import numpy as np

from .utils import get_random_tag, NpShmemArray

class SharedReplayBuffer():

    def __init__(self, 
        max_replay_buffer_size,
        task_nums
    ):
        self._max_replay_buffer_size = max_replay_buffer_size
        self.task_nums = task_nums

        assert self._max_replay_buffer_size % self.task_nums == 0, \
            "buffer size is not dividable by worker num"
        self._max_replay_buffer_size //= self.task_nums

        if not hasattr(self, "tag"):
            self.tag = get_random_tag()

    def build_by_example(self, example_dict):
        self._size = NpShmemArray(1, np.int32, self.tag+"_size")
        self._top  = NpShmemArray(1, np.int32, self.tag+"_top")

        self.tags = {}
        self.shapes = {}
        self.types = {}
        for key in example_dict:
            if not hasattr( self, "_" + key ):
                current_tag = "_"+key
                self.tags[current_tag] = self.tag+current_tag
                shape = (self._max_replay_buffer_size, self.task_nums) + \
                    np.shape(example_dict[key][0])
                self.shapes[current_tag] = shape
                self.types[current_tag] = example_dict[key][1]
                
                np_array = NpShmemArray(shape, example_dict[key][1], self.tag+current_tag)
                self.__setattr__(current_tag, np_array)

    def rebuild_from_tag(self):
        self._size  = NpShmemArray(1, np.int32, 
            self.tag+"_size", create=False)
        self._top   = NpShmemArray(1, np.int32,
            self.tag+"_top", create=False)

        for key in self.tags:
            np_array = NpShmemArray(self.shapes[key], self.types[key],
                self.tags[key], create=False)
            self.__setattr__(key, np_array )

    def add_sample(self, sample_dict):
        for key in sample_dict:
            self.__getattribute__( "_" + key )[self._top[0], :] = sample_dict[key]
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top[0] = (self._top[0] + 1) % \
            self._max_replay_buffer_size
        if self._size[0] < self._max_replay_buffer_size:
            self._size[0] = self._size[0] + 1

    def random_batch(self, batch_size, sample_key, reshape = True):
        assert batch_size % self.task_nums == 0, \
            "batch size should be dividable by task_nums"
        batch_size //= self.task_nums
        size = self.num_steps_can_sample()
        
        indices = np.random.randint(0, size, batch_size)
        return_dict = {}
        for key in sample_key:
            return_dict[key] = self.__getattribute__("_"+key)[indices]
            if reshape:
                return_dict[key] = return_dict[key].reshape(
                    (batch_size * self.task_nums, -1))
        return return_dict

    def num_steps_can_sample(self):
        # Use asynchronized sampling could cause sample collected is 
        # different across different workers but actually it's find
        min_size = np.min(self._size)
        return min_size