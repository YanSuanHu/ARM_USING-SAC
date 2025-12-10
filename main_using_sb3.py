import os
import numpy as np
import torch
import random

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from make_env import make_env

if __name__ == "__main__":
    # 一些超参数
    seed = 42
    num_envs = 8
    total_timesteps = 10000000

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    visuable = False

    # =========================================
    # 1. 创建并行环境
    # =========================================
    # SubprocVecEnv需要传入一个环境构造函数列表
    env_fns = [make_env(seed, rank=i,visuable=visuable) for i in range(num_envs)]
    # env_fns.append(make_env(seed,rank=7,visuable=True))
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    # =========================================
    # 2. 创建Stable-Baselines3的 SAC模型
    # =========================================
    # 可以在此处指定超参数，例如learning_rate、buffer_size、batch_size等等
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./sac_tensorboard_logs/",
        learning_rate=3e-4,  # 对应actor_lr/critic_lr
        buffer_size=1000000,  # 对应您的Replay Buffer大小
        batch_size=1024,  # 采样批次大小
        tau=0.005,  # 软更新系数
        gamma=0.99,  # 折扣因子
        ent_coef='auto',   # 如果想让alpha自动学习，可设置'ent_coef="auto"'
        seed=seed,
        policy_kwargs = dict(
        net_arch=dict(pi =[512, 512, 512],
                      qf =[512,512,512])
    ))

    # =========================================
    # 3. 开始训练
    # =========================================\
    try:
        print("开始训练... 按下 Ctrl+C 可以中途停止并保存模型。")
        model.learn(total_timesteps=total_timesteps)

    except KeyboardInterrupt:
        print("训练被手动中断。")
    finally:
        model.save("sac_armEnv_parallel_1")
        model.save_replay_buffer("sac_armEnv_parallel_1_replay_buffer")


    # 训练结束后关闭环境
    vec_env.close()
    print("Training finished and model saved!")
