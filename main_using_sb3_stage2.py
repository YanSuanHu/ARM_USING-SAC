import os
import numpy as np
import torch
import random

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from make_env import make_env

if __name__ == "__main__":
    # 一些超参数
    seed = 42
    num_envs = 8
    # 额外要训练的步数
    additional_timesteps = 4000000

    # 模型文件的路径
    model_path = "sac_armEnv_parallel_1"
    new_model_save_path = "sac_armEnv_parallel_2"  # 新模型保存路径

    # 设置随机种子 (如果需要复现)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    visuable = False

    # =========================================
    # 1. 像之前一样，创建并行环境
    #    加载模型时需要传入环境实例
    # =========================================
    env_fns = [make_env(seed, rank=i, visuable=visuable) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # =========================================
    # 2. 加载已保存的 SAC 模型
    # =========================================
    # 使用 SAC.load() 方法加载模型
    # 必须将环境实例传递给 `env` 参数
    print(f"从 {model_path}.zip 加载模型...")
    model = SAC.load(model_path, env=vec_env)
    try:
        print(f"加载 Replay Buffer...")
        model.load_replay_buffer(f"{model_path}_replay_buffer.pkl")
    except FileNotFoundError:
        print("未找到 Replay Buffer 文件，将使用新的空 Buffer。")

    # 如果想在加载后更改某些参数 (例如学习率)，可以这样做:
    # model.learning_rate = 1e-4

    # =========================================
    # 3. 在原有基础上继续训练
    # =========================================
    try:
        print("继续训练... 按下 Ctrl+C 可以中途停止并保存模型。")
        # 调用 learn 方法，它会从上次停止的地方继续
        # 注意：total_timesteps 是指这次要运行的额外步数
        model.learn(total_timesteps=additional_timesteps,
                    reset_num_timesteps=False,  # 关键参数！确保时间步是连续计数的
                    tb_log_name="SAC_continue")  # 建议为Tensorboard起个新名字

    except KeyboardInterrupt:
        print("训练被手动中断。")
    finally:
        # 强烈建议保存到新的文件名，以防覆盖掉之前的模型
        model.save(new_model_save_path)

    # 训练结束后关闭环境
    vec_env.close()
    print(f"Training finished and new model saved to {new_model_save_path}.zip!")