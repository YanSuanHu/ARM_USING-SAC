import torch
import multiprocessing as mp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC,PPO,A2C,TD3
from make_env import make_env
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
    env = SubprocVecEnv([make_env(seed=0,rank=i, visuable=False) for i in range(num_cpu)])
    lr_schedule = cosine_schedule(initial_value=0.0003, final_value=1e-6)

    # model = A2C(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="./a2c_tensorboard_logs/",
    #     learning_rate=lr_schedule,
    #     device=device  # Pass the explicitly determined device
        
    # )

    # print(f"--- Final check: Model is on device: {model.device} ---")
    # model.learn(total_timesteps=2000000)
    
    # # Save the model
    # model.save("a2c_armEnv_parallel_final")

    # model3 = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="./dqn_tensorboard_logs/",
    #     learning_rate=lr_schedule,
    #     device=device  # Pass the explicitly determined device
    # )
    # model3.learn(total_timesteps=2000000)
    # model3.save("dqn_armEnv_parallel_final")

    model2 = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./td3_tensorboard_logs/",
        learning_rate=lr_schedule,
        device=device  # Pass the explicitly determined device
    )
    model2.learn(total_timesteps=2000000)
    model2.save("td3_armEnv_parallel_final")


if __name__ == "__main__":
    print("--- Starting training ---")
    main()