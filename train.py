import torch
import multiprocessing as mp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC,PPO,A2C,TD3
from make_env import make_env
from make_env_copy import make_env as make_env_copy
import math
from typing import Callable

def cosine_schedule(initial_value: float, final_value: float = 1e-6) -> Callable[[float], float]:
    """
    余弦学习率退火.

    :param initial_value: 初始学习率
    :param final_value: 最终学习率
    :return: schedule a function that takes progress_remaining and returns the learning rate
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        # (1 - progress_remaining) 从 0 线性增长到 1
        # math.cos(...) 的值会从 cos(0)=1 平滑过渡到 cos(pi)=-1
        # 最终学习率会从 initial_value 平滑过渡到 final_value
        return final_value + 0.5 * (initial_value - final_value) * (1 + math.cos(math.pi * (1 - progress_remaining)))

    return func

def main():
    print('开始训练！')
    # Set the start method for multiprocessing to ensure GPU context is passed correctly
    try:
        mp.set_start_method("spawn", force=True)
        print("--- Multiprocessing start method set to 'spawn' ---")
    except RuntimeError:
        print("--- Multiprocessing start method already set ---")

    # Explicitly determine and set the device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("--- MPS is available! Setting device to 'mps'. ---")
    else:
        device = "cpu"
        print("--- MPS not available. Setting device to 'cpu'. ---")

    num_cpu = 8  # Number of parallel environments
    env1 = SubprocVecEnv([make_env(seed=0,rank=i, visuable=False) for i in range(num_cpu)])
    lr_schedule = cosine_schedule(initial_value=0.0003, final_value=1e-6)



    model4 = SAC(
        "MlpPolicy",
        env1,
        verbose=1,
        tensorboard_log="./sac_tensorboard_logs_without_IK/",
        learning_rate=lr_schedule,
        device=device  # Pass the explicitly determined device
    )
    model4.learn(total_timesteps=4000000)
    model4.save("sac_armEnv_parallel_final_without_IK")
    env1.close()

    # env2 = SubprocVecEnv([make_env_copy(seed=0,rank=i, visuable=False) for i in range(num_cpu)])
    # model5 = SAC(
    #     "MlpPolicy",
    #     env2,
    #     verbose=1,
    #     tensorboard_log="./sac_tensorboard_logs_with_IK/",
    #     learning_rate=lr_schedule,
    #     device=device  # Pass the explicitly determined device
    # )
    # model5.learn(total_timesteps=4000000)
    # model5.save("sac_armEnv_parallel_final_with_IK")


if __name__ == "__main__":
    print("消融实验训练SAC模型第二次")
    main()