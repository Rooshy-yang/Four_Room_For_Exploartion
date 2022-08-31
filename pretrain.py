import numpy as np

from four_room import Four_Rooms_Environment
from agent.ourc import OURCAgent
from agent.diayn import DIAYN
from replay_buffer import ReplayBuffer
import hydra
import torch


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.agent = make_agent(cfg.obs_type,
                                np.array([1]),
                                np.array([1]),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

    def train(self):
        global_step = 0
        buffer = ReplayBuffer(obs_dim=1, act_dim=1, skill_dim=16, size=2000000)
        env = Four_Rooms_Environment()
        timestep = env.reset()
        obs = timestep['observation']
        meta = self.agent.init_meta()
        while global_step < self.cfg.num_train_frames:

            action = self.agent.act(obs, meta['skill'])
            next_obs, reward, done,_ = env.step(action)
            next_meta = self.agent.update_meta(meta, global_step, obs)
            buffer.store(obs, action, reward, next_obs, done, meta['skill'], next_meta['skill'])
            meta = next_meta
            if done:
                timestep = env.reset()
                obs = timestep['observation']
            else:
                obs = next_obs

            if global_step > self.cfg.num_seed_frames:
                self.agent.update(buffer, global_step)

            global_step += 1


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
