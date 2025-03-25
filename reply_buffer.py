import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store_transition(self, s, a, r, s_, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s, a, r, s_, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        s, a, r, s_, done = zip(*batch)
        return {
            'states': np.array(s),
            'actions': np.array(a),
            'rewards': np.array(r),
            'next_states': np.array(s_),
            'dones': np.array(done)
        }

    def size(self):
        return len(self.buffer)
