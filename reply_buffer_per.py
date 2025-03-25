import numpy as np


class ReplayBufferPER:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-6):

        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon

    def store_transition(self, s, a, r, s_, done,is_expert =False):
        """保存一个 transition，同时为其设置初始优先级为当前已有优先级中的最大值，或初始默认值1."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s, a, r, s_, done)
        # 设置当前 transition 的初始优先级
        if is_expert:
            self.priorities[self.position] = 1000.0
        else:
            max_priority = self.priorities.max() if self.buffer and self.priorities.max() > 0 else 1.0
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """根据优先级采样，并返回采样 transition、对应索引以及重要性采样权重"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        # 计算每个样本的采样概率
        probs = prios ** self.alpha
        probs /= probs.sum()

        # 根据概率分布采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        s, a, r, s_, done = zip(*batch)

        # 计算重要性采样权重： w_i = (N * P(i))^(-beta)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        # 为了稳定性，将权重归一化（最大值设为1）
        weights /= weights.max()

        # 随着采样次数增加，逐步增大 beta（最终趋向于1）
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        return {
            'states': np.array(s),
            'actions': np.array(a),
            'rewards': np.array(r),
            'next_states': np.array(s_),
            'dones': np.array(done)
        }, indices, weights

    def update_priorities(self, indices, priorities):
        """根据新的 TD 误差更新采样样本的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def size(self):
        return len(self.buffer)
