

from Env import armEnv
from RL import SAC
from RL import ReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

MAX_EPISODES = 500
MAX_EP_STEPS = 200

env = armEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound
actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 500
batch_size = 64
target_entropy = -1
reward_list =[]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

rl = SAC(s_dim, hidden_dim, a_dim, a_bound,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device)
replay_buffer = ReplayBuffer(buffer_size)

for episode in range(MAX_EPISODES):
    s = env.reset()  # 重置环境，获得初始状态
    ep_reward = 0
    for step in range(MAX_EP_STEPS):
        a = rl.take_action(s)     # agent 选择动作
        s_next, r, done = env.step(a)    # 在环境中执行动作，获得下一个状态、奖励和结束标志

        # 将转移存入经验回放缓冲区
        replay_buffer.store_transition(s, a, r, s_next, done)
        ep_reward += r

        # 当缓冲区中样本足够时，采样批次数据并更新 agent
        if replay_buffer.size() >= minimal_size:
            batch = replay_buffer.sample(batch_size)
            rl.update(batch)
        if done:
            break
        s = s_next
    reward_list.append(ep_reward)
    print("Episode: {}, Reward: {}".format(episode, ep_reward))
env.close()
plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.show()
torch.save({
    'actor': rl.actor.state_dict(),
    'critic1': rl.critic_1.state_dict(),
    'critic2': rl.critic_2.state_dict(),
    'target_critic1': rl.target_critic_1.state_dict(),
    'target_critic2': rl.target_critic_2.state_dict(),
    'log_alpha': rl.log_alpha
}, 'sac_checkpoint.pth')

