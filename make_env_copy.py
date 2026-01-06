
from anotherArm_copy import UR5RobotiqEnv as armEnv  # 您的自定义环境
from stable_baselines3.common.monitor import Monitor
def make_env(seed, rank=0,visuable=False):

    def _init():
        env = armEnv(visuable)
        # 如果env有seed方法，就调用一下确保可复现
        # env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init
