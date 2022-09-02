import time

import gym
import numpy as np
from fourroom import FourRoom
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
        self.env = FourRoom()

        self.agent = make_agent(cfg.obs_type,
                                self.env.observation_space,
                                self.env.action_space,
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

    def train(self):
        start = time.time()
        global_step = 0
        buffer = ReplayBuffer(obs_dim=self.env.observation_space.shape[0], act_dim=self.env.action_space.shape[0],
                              skill_dim=16, size=10000)
        # only position
        obs = self.env.reset()
        meta = self.agent.init_meta()
        state_buffer = np.zeros([self.cfg.num_train_frames, self.env.observation_space.shape[0]])
        while global_step < self.cfg.num_train_frames:
            with torch.no_grad():
                action = self.agent.act(obs, meta, global_step, False)
            state_buffer[global_step] = obs
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = next_obs
            # print(obs, action, np.argmax(meta['skill']), next_obs, done, global_step)
            next_meta = self.agent.update_meta(meta, global_step, obs)
            # buffer.store(obs, action, reward, next_obs, done, meta['skill'], next_meta['skill'])
            buffer.store(obs, action, reward, next_obs, done,)
            meta = next_meta
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
            # if global_step > self.cfg.num_seed_frames:
            #     self.agent.update(buffer, global_step)

            global_step += 1
        end = time.time()
        print("time :", end - start)
        torch.save(self.agent, 'ourc.pkl')
        np.save('Q_table', self.agent.Q_table)
        np.save('state',state_buffer)


@hydra.main(config_path='.', config_name='continuous')
def main(cfg):
    from continuous import Workspace as W
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
