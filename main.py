

from env import armEnv
from rl_algorithm import SAC
from reply_buffer import ReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
# from test_env import TestArmEnv

MAX_EPISODES = 500
MAX_EP_STEPS = 220


env = armEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound
actor_lr = 1e-3
critic_lr = 1e-3
alpha_lr = 1e-2
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 500
batch_size = 64
target_entropy = -float(a_dim)
reward_list =[]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
rl = SAC(s_dim, hidden_dim, a_dim, a_bound,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device)

demo_data = np.load("master_data.npz")
obs_demo = demo_data["obs"]
act_demo = demo_data["act"]
next_obs_demo = demo_data["obs_next"]
rewards_demo = demo_data["rewards"]
done_demo = demo_data["done"]

replay_buffer = ReplayBuffer(buffer_size)
num_episodes, episode_length, _ = obs_demo.shape
for i in range(num_episodes):
    for j in range(episode_length):
        replay_buffer.store_transition(
            obs_demo[i, j],
            act_demo[i, j],
            rewards_demo[i, j],
            next_obs_demo[i, j],
            done_demo[i, j]
        )

for episode in range(MAX_EPISODES):
    s = env.reset()  # 重置环境，获得初始状态
    ep_reward = 0
    for step in range(MAX_EP_STEPS):
        a = rl.take_action(s)     # agent 选择动作
        s_next, r, done = env.step(a)    # 在环境中执行动作，获得下一个状态、奖励和结束标志
        if done:
            print("第",episode,"个epsido，第",step,"个动作，完成目标")

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

