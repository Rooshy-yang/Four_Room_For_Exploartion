import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, skill_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.skill_buf = np.zeros(combined_shape(size, skill_dim), dtype=np.float32)
        self.skill2_buf = np.zeros(combined_shape(size, skill_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, meta=None, next_meta=None):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if meta is not None and next_meta is not None:
            for value in meta.values():
                self.skill_buf[self.ptr] = value
            for value in next_meta.values():
                self.skill2_buf[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     skill=self.skill_buf[idxs],
                     nextskill=self.skill2_buf[idxs])
        return batch

    def save(self):
        data = (self.obs_buf, self.obs2_buf, self.act_buf, self.skill_buf, self.skill2_buf, self.rew_buf, self.done_buf)
        for i, value in enumerate(data):
            np.save(str(i), value)
