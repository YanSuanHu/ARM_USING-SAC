import random
import pybullet as p
import pybullet_data  # pybullet 自带的一些模型
import os
import numpy as np
import gym


class TestArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.physicsClient = p.connect(p.GUI)  # 或者 p.DIRECT
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # 设置重力
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        self.robotStartPos = [0, 0, 0]
        self.robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.setAdditionalSearchPath(os.path.abspath("models"))
        self.robotId = p.loadSDF("kuka_with_gripper.sdf")  # 加载机械臂
        p.resetBasePositionAndOrientation(self.robotId[0], self.robotStartPos, self.robotStartOrientation)
        self.action_dim = 6
        self.distance_threshold = 0.05
        rand_x = random.uniform(-0.5,0.5)
        self.position = [rand_x,0.5,0.32]
        self.end_effector_link_index = 7
        self.state_dim = 6
        self.action_bound = [1]*self.action_dim
        s = self.get_obs()  # 获取初始状态
        self.origin_dis = np.linalg.norm(s[:3] - self.position)
        # self.origin_dis = np.linalg.norm(np.array(self.position) - np.array(self.robotStartPos), axis=-1)
        for i in range(3,6):
            self.action_bound[i] = np.pi


    def step(self, action):
        self.take_act(action)
        for _ in range(20):
            p.stepSimulation()
        obs = self.get_obs()
        done = False
        done,reward = self.compute_reward(obs[:3], self.position)
        return obs, reward, done

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # 设置重力
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        self.robotStartPos = [0, 0, 0]
        self.robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.setAdditionalSearchPath(os.path.abspath("models"))
        self.robotId = p.loadSDF("kuka_with_gripper.sdf")  # 加载机械臂
        p.resetBasePositionAndOrientation(self.robotId[0], self.robotStartPos, self.robotStartOrientation)
        self.action_dim = 6
        self.distance_threshold = 0.05
        rand_x = random.uniform(-0.5,0.5)
        self.position = [rand_x,0.5,0.32]
        self.end_effector_link_index = 7
        self.state_dim = 6
        self.action_bound = [1]*self.action_dim
        s = self.get_obs()  # 获取初始状态
        self.origin_dis = np.linalg.norm(s[:3] - self.position)
        # self.origin_dis = np.linalg.norm(np.array(self.position) - np.array(self.robotStartPos), axis=-1)
        for i in range(3,6):
            self.action_bound[i] = np.pi
        obs = self.get_obs()
        return obs

    def compute_reward(self, achieved_goal, goal):
        achieved_goal = np.array(achieved_goal)
        goal = np.array(goal)
        assert achieved_goal.shape == goal.shape
        dis =np.linalg.norm(achieved_goal - goal, axis=-1)
        flag = dis < self.distance_threshold
        if flag:
            reward = 200
        else:
            reward = self.origin_dis-dis
        return [flag,reward]


    def get_obs(self):
        state = p.getLinkState(self.robotId[0], self.end_effector_link_index)
        pos = np.array(state[4])
        # orientation = np.array(p.getEulerFromQuaternion(state[5]))
        # linear_v = np.array(state[6])
        # angular_v = np.array(state[7])
        target = np.array(self.position)
        obs = np.concatenate([pos,target])
        return obs

    def take_act(self,Commands):

        pos = p.getLinkState(self.robotId[0], self.end_effector_link_index)[4]
        new_pos = [pos[0]+Commands[0],pos[1]+Commands[1],pos[2]+Commands[2]]
        target_orientation = p.getQuaternionFromEuler(Commands[3:6])
        targetPositionsJoints = p.calculateInverseKinematics(
            self.robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
            self.end_effector_link_index,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
            new_pos,  # 目标末端位置
            target_orientation
        )
        p.setJointMotorControlArray(
            bodyUniqueId=self.robotId[0],  # 机械臂ID
            jointIndices=range(self.end_effector_link_index),  # 需要控制的关节索引
            controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
            targetPositions=targetPositionsJoints[0:self.end_effector_link_index]  # IK 求解得到的目标关节角
        )

    def close(self):
        if self.physicsClient is not None:
            p.disconnect(self.physicsClient)
            self.physicsClient = None



