import time

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
                                self.env.observation_space.shape[0],
                                self.env.action_space.shape[0],
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

    def train(self):
        start = time.time()
        global_step = 0
        buffer = ReplayBuffer(obs_dim=self.env.observation_space.shape[0], act_dim=self.env.action_space.shape[0],
                              skill_dim=16, size=10000)
        # only position
        obs = self.env.reset()[0]
        meta = self.agent.init_meta()
        state_buffer = np.zeros(self.cfg.num_train_frames)
        while global_step < self.cfg.num_train_frames:
            action = self.agent.act(obs, meta['skill'])
            state_buffer[global_step] = obs
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = next_obs[0]
            # print(obs, action, np.argmax(meta['skill']), next_obs, done, global_step)
            next_meta = self.agent.update_meta(meta, global_step, obs)
            buffer.store(obs, action, reward, next_obs, done, meta['skill'], next_meta['skill'])
            meta = next_meta
            if done:
                obs = self.env.reset()[0]
            else:
                obs = next_obs
            # if global_step > self.cfg.num_seed_frames:
            #     self.agent.update(buffer, global_step)

            global_step += 1
        end = time.time()
        print("time :",end - start)
        torch.save(self.agent, 'ourc.pkl')
        np.save('Q_table', self.agent.Q_table)
        np.save('state',state_buffer)


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
