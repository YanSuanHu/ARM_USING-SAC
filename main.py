

from env import armEnv
from rl_algorithm import SAC
from reply_buffer_per import ReplayBufferPER
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
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 0.2
hidden_dim = 256
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 500
batch_size = 256
target_entropy = -float(a_dim)
reward_list =[]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
rl = SAC(s_dim, hidden_dim, a_dim, a_bound,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device)

demo_data = np.load("master_data_dense.npz",allow_pickle=True)
obs_demo = demo_data["obs"]
act_demo = demo_data["act"]
next_obs_demo = demo_data["obs_next"]
rewards_demo = demo_data["rewards"]
done_demo = demo_data["done"]

replay_buffer = ReplayBufferPER(buffer_size)
num_episodes = len(obs_demo)

for i in range(num_episodes):
    # 对于每个 episode，取出对应的所有转移数据
    episode_obs = obs_demo[i]
    episode_act = act_demo[i]
    episode_rewards = rewards_demo[i]
    episode_next_obs = next_obs_demo[i]
    episode_done = done_demo[i]
    # 对于该 episode 中的每个时间步（transition）
    for j in range(len(episode_obs)):
        replay_buffer.store_transition(
            episode_obs[j],
            episode_act[j],
            episode_rewards[j],
            episode_next_obs[j],
            episode_done[j],
            is_expert = True
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
            # PER replay buffer 需要返回采样数据、样本索引和重要性采样权重
            batch, indices, is_weights = replay_buffer.sample(batch_size)
            # 将 is_weights 转换为 torch tensor，并发送到 device
            is_weights = torch.tensor(is_weights, dtype=torch.float).to(device)
            # 更新网络，update 返回的 TD 误差用于更新优先级
            td_errors = rl.update(batch, is_weights)
            replay_buffer.update_priorities(indices, td_errors)
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

