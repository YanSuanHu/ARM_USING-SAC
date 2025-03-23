import random
import pybullet as p
import pybullet_data  # pybullet 自带的一些模型
import os
import numpy as np
import gym


class armEnv(gym.Env):
    def __init__(self):
        physicsClient = p.connect(p.GUI)  # 或者 p.DIRECT
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # 设置重力
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        self.robotStartPos = [0, 0, 0]
        self.robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.setAdditionalSearchPath(os.path.abspath("models"))
        self.robotId = p.loadSDF("kuka_with_gripper.sdf")  # 加载机械臂
        p.resetBasePositionAndOrientation(self.robotId[0], self.robotStartPos, self.robotStartOrientation)
        self.action_dim = 7
        self.distance_threshold = 0.05
        rand_x = random.uniform(-0.5,0.5)
        self.tableId = p.loadURDF("table_square/table_square.urdf",[rand_x,1.,0],globalScaling=0.5)
        self.colaId = p.loadURDF("Household-items-urdfs/urdf/cola.urdf",[rand_x,1,0.32])
        self.position = [rand_x,1,0.32]
        self.target_position = [0, -1, 0]
        self.end_effector_link_index = 7
        self.state_dim = 15
        self.action_bound = [1]*self.action_dim
        for i in range(4,7):
            self.action_bound[i] = np.pi/2
        p.changeDynamics(bodyUniqueId=self.colaId,
                         linkIndex=-1,  # 对 base 使用 -1
                         lateralFriction=5.0,
                         spinningFriction=1.0,
                         rollingFriction=0.001)


    def step(self, action):
        self.take_act(action)
        for _ in range(20):
            p.stepSimulation()
        obs = self.get_obs()
        done = False
        done,reward = self.compute_reward(obs[-3:], self.target_position)
        return obs, reward, done

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")
        p.setAdditionalSearchPath(os.path.abspath("models"))
        self.robotId = p.loadSDF("kuka_with_gripper.sdf")
        p.resetBasePositionAndOrientation(self.robotId[0],
                                          self.robotStartPos,
                                          self.robotStartOrientation)
        rand_x = random.uniform(-0.5, 0.5)
        self.tableId = p.loadURDF("table_square/table_square.urdf", [rand_x, 1, 0], globalScaling=0.5)
        self.colaId = p.loadURDF("Household-items-urdfs/urdf/cola.urdf", [rand_x, 1, 0.32])
        self.position = [rand_x, 1, 0.32]
        self.target_position = [0, -1, 0]
        p.changeDynamics(bodyUniqueId=self.colaId,
                         linkIndex=-1,  # 对 base 使用 -1
                         lateralFriction=5.0,
                         spinningFriction=1.0,
                         rollingFriction=0.001)
        obs = self.get_obs()
        return obs

    def compute_reward(self, achieved_goal, goal):
        achieved_goal = np.array(achieved_goal)
        goal = np.array(goal)
        assert achieved_goal.shape == goal.shape
        dis =np.linalg.norm(achieved_goal - goal, axis=-1)
        if dis>self.distance_threshold:
            return [False,-1]
        else:
            return [True,0]

    def get_obs(self):
        state = p.getLinkState(self.robotId[0], self.end_effector_link_index,computeLinkVelocity=1)
        pos = np.array(state[4])
        orientation = np.array(p.getEulerFromQuaternion(state[5]))
        linear_v = np.array(state[6])
        angular_v = np.array(state[7])
        cola_position = np.array(p.getBasePositionAndOrientation(self.colaId)[0])
        obs = np.concatenate([pos, orientation, linear_v, angular_v,cola_position])
        return obs

    def take_act(self,Commands):
        def close_gripper(robot_id):
            # 将夹爪闭合到接近 0
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=8,
                controlMode=p.POSITION_CONTROL,
                targetPosition=-0.05,
                force=5000
            )
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=11,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.05,
                force=5000
            )
        def open_gripper(robot_id):
            # 将夹爪张开到一定角度
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=8,  # 左爪根部
                controlMode=p.POSITION_CONTROL,
                targetPosition=-0.4,  # 角度(或弧度)，可自行调整
                force=50  # 能施加的力/扭矩上限
            )
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=11,  # 右爪根部
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.4,  # 往另一个方向打开
                force=50
            )
        pos = p.getLinkState(self.robotId[0], self.end_effector_link_index)[4]
        new_pos = [pos[0]+Commands[0],pos[1]+Commands[1],pos[2]+Commands[2]]
        target_orientation = p.getQuaternionFromEuler(Commands[3:6])
        targetPositionsJoints = p.calculateInverseKinematics(
            self.robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
            self.end_effector_link_index,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
            new_pos,  # 目标末端位置
            target_orientation
        )
        if Commands[-1]==1:
            close_gripper(self.robotId[0])
        if Commands[-1]==-1:
            open_gripper(self.robotId[0])
        p.setJointMotorControlArray(
            bodyUniqueId=self.robotId[0],  # 机械臂ID
            jointIndices=range(self.end_effector_link_index),  # 需要控制的关节索引
            controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
            targetPositions=targetPositionsJoints[0:self.end_effector_link_index]  # IK 求解得到的目标关节角
        )



