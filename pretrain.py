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
        buffer = ReplayBuffer(obs_dim=1, act_dim=1, size=2000000)
        env = Four_Rooms_Environment()
        obs = env.reset()
        meta = self.agent.init_meta()

        while global_step < self.cfg.num_train_frames:

            with torch.no_grad():
                action = self.agent.act(obs, meta, global_step, eval_mode=False)

            next_obs, reward, done = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            if done:
                obs = env.reset()
            else:
                obs = next_obs
            self.agent.update_meta(meta, global_step, obs)

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
