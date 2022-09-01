import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.sarsa import Sarsa


class GeneratorB(nn.Module):
    def __init__(self, tau_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(tau_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, tau):
        skill_pred = self.skill_pred_net(tau)
        return skill_pred


class Discriminator(nn.Module):
    def __init__(self, tau_dim, feature_dim, hidden_dim):
        super().__init__()
        # def SimClR :
        self.embed = nn.Sequential(nn.Linear(tau_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, feature_dim))

        self.project_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, feature_dim))
        self.apply(utils.weight_init)

    def forward(self, tau):
        features = self.embed(tau)
        features = self.project_head(features)
        return features


class OURCAgent(Sarsa):
    def __init__(self, update_skill_every_step, contrastive_scale,
                 update_encoder, contrastive_update_rate, temperature, update_every_steps, **kwargs):
        super().__init__(**kwargs)
        self.lr = kwargs['lr']
        self.update_skill_every_step = update_skill_every_step
        self.contrastive_scale = contrastive_scale
        self.update_encoder = update_encoder
        self.batch_size = kwargs['batch_size']
        self.contrastive_update_rate = contrastive_update_rate
        self.temperature = temperature
        self.device = kwargs['device']
        self.tau_len = update_skill_every_step
        self.update_every_steps = update_every_steps
        self.obs_dim = kwargs['obs_dim']
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create ourc
        self.gb = GeneratorB(self.obs_dim, self.skill_dim,
                             kwargs['hidden_dim']).to(kwargs['device'])

        self.discriminator = Discriminator(self.obs_dim,
                                           self.skill_dim,
                                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.gb_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = nn.CrossEntropyLoss()

        # optimizers
        self.gb_opt = torch.optim.Adam(self.gb.parameters(), lr=self.lr)
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.gb.train()
        self.discriminator.train()

        self.skill_ptr = 0

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[self.skill_ptr] = 1
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if global_step % self.update_skill_every_step == 0:
            self.skill_ptr = (self.skill_ptr + 1) % self.skill_dim
            return self.init_meta()
        return meta

    def update_gb(self, skill, gb_batch, step):
        metrics = dict()
        labels = torch.argmax(skill, dim=1)
        loss, df_accuracy = self.compute_gb_loss(gb_batch, labels)

        self.gb_opt.zero_grad()
        loss.backward()
        self.gb_opt.step()

        return metrics

    def update_contrastive(self, taus, skills):
        metrics = dict()
        features = self.discriminator(taus)
        loss = self.compute_info_nce_loss(features, skills)
        loss = torch.mean(loss)

        self.dis_opt.zero_grad()
        loss.backward()
        self.dis_opt.step()

        return metrics

    def compute_intr_reward(self, skills, tau_batch, metrics):

        # compute q(z | tau) reward
        d_pred = self.gb(tau_batch)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        gb_reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]), torch.argmax(skills, dim=1)] - math.log(
            1 / self.skill_dim)
        gb_reward = gb_reward.reshape(-1, 1)

        # compute contrastive reward
        features = self.discriminator(tau_batch)

        # maximize softmax item
        contrastive_reward = torch.exp(-self.compute_info_nce_loss(features, skills))
        intri_reward = gb_reward + contrastive_reward * self.contrastive_scale
        return intri_reward

    def compute_info_nce_loss(self, features, skills):

        size = features.shape[0] // self.skill_dim

        labels = torch.argmax(skills, dim=1)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)

        # don't limit update for all negative
        pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
        pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)
        neg = (~labels.bool()).long()
        # select one and combine multiple positives
        positives = torch.sum(similarity_matrix * pick_one_positive_sample_idx, dim=-1, keepdim=True)
        negatives = torch.sum(similarity_matrix * neg, dim=-1, keepdim=True)

        loss = -torch.log(positives / negatives)

        return loss

    def compute_gb_loss(self, taus, skill):
        """
        DF Loss
        """

        d_pred = self.gb(taus)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.gb_criterion(d_pred, skill)
        df_accuracy = torch.sum(
            torch.eq(skill,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
            pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = buffer.sample_batch(32)
        obs, next_obs, action, rew, done, skill, next_skill = batch.values()
        metrics.update(self.update_contrastive(next_obs, skill))

        # update q(z | tau)
        # bucket count for less time spending
        metrics.update(self.update_gb(skill, next_obs, step))

        # compute intrinsic reward
        with torch.no_grad():
            intr_reward = self.compute_intr_reward(skill, next_obs, metrics)

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        action = action.numpy().astype('int').flatten()
        obs = obs.numpy().astype('int').flatten()
        next_obs = next_obs.numpy().astype('int').flatten()
        next_action = self.act(next_obs, skill).flatten()
        skill = torch.argmax(skill, dim=1).numpy()
        intr_reward = intr_reward.numpy().flatten()
        td_error = intr_reward + self.gamma * self.Q_table[next_obs, skill, next_action] - \
                   self.Q_table[obs, skill, action]
        self.Q_table[obs, skill, action] += self.alpha * td_error
        return metrics
