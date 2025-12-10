import pybullet as p
import pybullet_data
import gym
import numpy as np
import os
import random
import time
from stable_baselines3 import SAC
from make_env import make_env


# ================================================================= #
#  1. 定义您的环境类 (与训练时相同, 仅修改一行以启动GUI)
# ================================================================= #

if __name__ == "__main__":
    # 指定要加载的模型路径
    MODEL_PATH = "sac_armEnv_parallel_1.zip"

    print(f"正在加载模型: {MODEL_PATH}")

    # 创建带GUI的环境实例
    env_make = make_env(seed=42,visuable=True)
    env = env_make()

    # 加载训练好的SAC模型
    try:
        model = SAC.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。请确保文件路径正确。")
        env.close()
        exit()

    # 循环测试10个回合
    num_episodes = 10
    for i in range(num_episodes):
        print(f"--- 开始第 {i + 1}/{num_episodes} 回合测试 ---")
        obs,info = env.reset()
        done = False
        total_reward = 0

        # 在一个回合内循环
        while not done:
            # model.predict() 会根据当前观测(obs)输出最佳动作
            # deterministic=True 表示我们用确定的策略（不再进行随机探索），这在评估时是标准做法
            action, _states = model.predict(obs, deterministic=True)

            # 在环境中执行动作
            obs, reward, terminated,truncated, info = env.step(action)
            total_reward += reward

            # 放慢仿真速度，以便肉眼观察 (PyBullet默认步长是1/240秒)
            time.sleep(1. / 240.)
            done = terminated or truncated

        print(f"回合结束。总奖励: {total_reward:.2f}")

    print("\n可视化测试完成。")
    # 关闭环境
    env.close()
