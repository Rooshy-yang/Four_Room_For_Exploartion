import numpy as np
import torch

class Sarsa:
    """ Sarsa算法 for skill"""

    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4, **kwargs):
        self.skill_dim = kwargs['skill_dim']
        self.Q_table = np.zeros([nrow * ncol, self.skill_dim, n_action])  # 初始化Q(s,z,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def act(self, state, skill):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if torch.is_tensor(skill):
            skill_num = torch.argmax(skill, dim=1).numpy()
        else:
            skill_num = np.argmax(skill)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action, size=state.shape[0])
        else:
            action = np.argmax(self.Q_table[state, skill_num], axis=-1)
        return action

    def best_action(self, state, skill):  # 用于打印策略
        skill_num = torch.argmax(skill, dim=1).item()
        Q_max = np.max(self.Q_table[state, skill_num])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def _update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
