import time

import numpy as np
import random
from env import armEnv

demo_num=10


def launch():
    env = armEnv()
    obs_dim = env.state_dim
    act_dim = env.action_dim
    max_timesteps = 220
    collect_demo_data(env,max_timesteps)

def collect_demo_data(env, max_timesteps,needed_success=demo_num):
    # 用于存储所有成功episode的数据
    obs_total = []
    actions_total = []
    next_obs_total = []
    rewards_total = []
    done_total = []

    success_count = 0
    max_episodes = 50000  # 最多尝试的episode数
    x_total = 0
    y_total = 0
    z_total = 0

    for episode_i in range(max_episodes):
        if success_count >= needed_success:
            break

        # 每个episode的临时存储
        ep_obs = []
        ep_acts = []
        ep_next_obs = []
        ep_rewards = []
        ep_done = []

        # 重置环境
        obs = env.reset()

        reward_total = 0
        old_pos = np.array(obs[:3]).astype(float)
        old_euler =np.array(obs[3:6]).astype(float)
        old_cola = np.array(obs[6:9]).astype(float)
        for t in range(max_timesteps):
            end_effector_pos = obs[:3]
            end_effector_euler = obs[3:6]
            cola_pos =obs[6:9]

            if t<=10:
                act = np.array([0, 0, 0, 0, 0, 0, -0.8])
            elif t<=90:
                dx = (old_cola[0] - old_pos[0])/80*(t-10)+old_pos[0]-end_effector_pos[0]
                dy = (old_cola[1]  - old_pos[1]-0.18)/80*(t-10)+old_pos[1]-end_effector_pos[1]
                dz = (old_cola[2] - old_pos[2]+0.2)/80*(t-10)+old_pos[2]-end_effector_pos[2]
                deulerx = (-np.pi/2-old_euler[0])/80*(t-10)+old_euler[0]-end_effector_euler[0]
                deulery = (-0 - old_euler[1]) / 80 * (t - 10) + old_euler[1] - end_effector_euler[1]
                deulerz = (-0 / 2 - old_euler[2]) / 80 * (t - 10) + old_euler[2] - end_effector_euler[2]
                act = np.array([dx, dy, dz, deulerx, deulery, deulerz, 0])
            elif t<=100:
                dx = 0
                dy = 0
                dz = -0.023
                deulerx = (-np.pi / 2 - new_euler[0]) / 80 * (t - 10) + new_euler[0] - end_effector_euler[0]
                deulery = (-0 - new_euler[1]) / 80 * (t - 10) + new_euler[1] - end_effector_euler[1]
                deulerz = (-0 / 2 - new_euler[2]) / 80 * (t - 10) + new_euler[2] - end_effector_euler[2]
                act = np.array([dx,dy,dz,deulerx,deulery,deulerz,0])
            elif t<=110:
                act = np.array([0,0,0,0,0,0,0.8])
            elif t <= 120:
                dx = 0
                dy = 0
                dz = 0.01
                act = np.array([dx,dy,dz,0,0,0,0])
            elif t<=160:
                dx =(old_pos[0]- new_pos[0])/40*(t-120)+new_pos[0]-end_effector_pos[0]
                dy = (old_pos[1]- new_pos[1])/40*(t-120)+new_pos[1]-end_effector_pos[1]
                dz = (old_pos[2] - new_pos[2])/40*(t - 120) + new_pos[2] - end_effector_pos[2]
                deulerx = (old_euler[0]-new_euler[0])/40*(t-120)+new_euler[0]-end_effector_euler[0]
                deulery = (old_euler[1]-new_euler[1])/40*(t-120)+new_euler[1]-end_effector_euler[1]
                deulerz = (old_euler[2] - new_euler[2]) / 40 * (t - 120) + new_euler[2] - end_effector_euler[2]
                act = np.array([dx,dy,dz,deulerx ,deulery ,deulerz,0])
            elif t<=200:
                dx = (0 - new_pos[0]-0.01)/40*(t-160)+new_pos[0]-end_effector_pos[0]
                dy = (-1 - new_pos[1]+0.2)/40*(t-160)+new_pos[1]-end_effector_pos[1]
                dz = (0 - new_pos[2]-0.06)/40*(t-160)+new_pos[2]-end_effector_pos[2]
                deulerx = (np.pi/2 - new_euler[0]) / 40 * (t - 160) + new_euler[0] - end_effector_euler[0]
                deulery = (0 - new_euler[1]) / 40 * (t - 160) + new_euler[1] - end_effector_euler[1]
                deulerz = (0 - new_euler[2]) / 40 * (t - 160) + new_euler[2] - end_effector_euler[2]
                act = np.array([dx,dy,dz,deulerx,deulery,deulerz,0])
            elif t<=210:
                deulerx = (np.pi / 2 - new_euler[0]) / 40 * (t - 160) + new_euler[0] - end_effector_euler[0]
                deulery = (0 - new_euler[1]) / 40 * (t - 160) + new_euler[1] - end_effector_euler[1]
                deulerz = (0 - new_euler[2]) / 40 * (t - 160) + new_euler[2] - end_effector_euler[2]
                act = np.array([0,0,0,deulerx,deulery,deulerz,0])
            else:
                act = np.array([0,0,0,0,0,0,-0.8])
            if t==90:
                new_pos = end_effector_pos
                new_euler = end_effector_euler
            if t ==120:
                new_pos = end_effector_pos
                new_euler = end_effector_euler
            if t ==160:
                new_pos = end_effector_pos
                new_euler = end_effector_euler
            if t==200:
                new_pos = end_effector_pos
                new_euler = end_effector_euler
            if t==219:
                print("第",episode_i+1," 次")
                # print("机械臂最终角度",end_effector_euler)
                # print("机械臂最终位置",end_effector_pos)
                # print("可乐最终位置",cola_pos)
                print("距离为",np.linalg.norm(np.array(cola_pos) - np.array([0,-1,0.08]), axis=-1))
                print("奖励为",reward_total)
                x_total += cola_pos[0]
                y_total += cola_pos[1]
                z_total +=cola_pos[2]
            ep_obs .append(obs.copy())
            ep_acts.append(act.copy())
            obs,reward,done = env.step(act)
            reward_total += reward
            ep_rewards.append(reward)
            ep_next_obs.append(obs.copy())
            ep_done.append(done)
            if done:
                success_count+=1
                obs_total.append(ep_obs)
                actions_total.append(ep_acts)
                next_obs_total.append(ep_next_obs)
                rewards_total.append(ep_rewards)
                done_total.append(ep_done)
                print(f"第 {episode_i + 1} 个 episode 成功，已收集成功示范 {success_count} 条。")
                print("奖励为", reward_total)
                break
    file = "master_data_dense.npz"
    print("完成！")
    # print(x_total/100)
    # print(y_total/100)
    # print(z_total/100)
    print(success_count)
    np.savez_compressed(
        file,
        act=np.array(actions_total, dtype=object),
        obs=np.array(obs_total, dtype=object),
        obs_next=np.array(next_obs_total, dtype=object),
        rewards=np.array(rewards_total, dtype=object),
        done=np.array(done_total, dtype=object)
    )
if __name__=="__main__":
    launch()