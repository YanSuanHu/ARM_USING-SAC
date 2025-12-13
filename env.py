import random
import pybullet as p
import pybullet_data  # pybullet 自带的一些模型
import os
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

class armEnv(gym.Env):
    def __init__(self,visualable):
        super().__init__()
        if visualable ==False:
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)
        p.resetSimulation()
        self.current_step =0
        self.max_episode_steps = 200

        self.action_dim = 5
        self.distance_threshold = 0.1

        self.end_effector_link_index = 7
        self.state_dim = 24
        low = np.array([-1, -1, -1, -1,-1], dtype=np.float32)
        high = np.array([ 1,  1,  1,  1,1], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
        self._seed = None
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        target_euler_angles = [-np.pi/2, 0, 0]
        self.target_orientation_quat = p.getQuaternionFromEuler(target_euler_angles)
        self.gripper_link_indices = [10, 13]
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 1. 在 init 中加载所有静态和动态物体
        self.plane_id = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("table_square/table_square.urdf", [0, 0.8, 0], globalScaling=0.5)
        # 先加载机器人和可乐，位置随便，reset时再归位
        p.setAdditionalSearchPath(os.path.abspath("models"))
        self.robotId = p.loadSDF("kuka_with_gripper.sdf")
        # 预加载 Cola，位置设在远处于避免碰撞
        self.colaId = p.loadURDF("Household-items-urdfs/urdf/cola.urdf", [0, 0, -10])
        self.robotStartPos = [0, 0, 0.5]
        self.robotStartOrientation = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])

    # def seed(self,seed):
    #     self._seed = seed
    #     np.random.seed(seed)
    #     random.seed(seed)
    # Place this inside your armEnv class
    def _calculate_robot_mass(self):
        total_mass = 0.0
        # Get mass of the base link
        base_dynamics = p.getDynamicsInfo(self.robotId[0], -1)
        total_mass += base_dynamics[0]  # mass is the first element

        # Get mass of all other links
        num_links = p.getNumJoints(self.robotId[0])
        for i in range(num_links):
            link_dynamics = p.getDynamicsInfo(self.robotId[0], i)
            total_mass += link_dynamics[0]
        return total_mass
    def step(self, action):
        self.current_step += 1
        self.take_act(action)
        # Apply the counter-gravity force before stepping the simulation
        # p.applyExternalForce(
        #     objectUniqueId=self.robotId[0],
        #     linkIndex=-1,  # Apply to the base link
        #     forceObj=self.counter_gravity_force,
        #     posObj=[0, 0, 0],  # Position doesn't matter for a pure force
        #     flags=p.WORLD_FRAME
        # )
        for _ in range(10):
            p.stepSimulation()
        obs = self.get_obs()
        is_success, reward = self.compute_reward(obs[6:9], self.target_position, self.prev_dist_to_goal)
        # is_success,reward = self.compute_reward(obs[6:9], self.target_position,obs[9:12])
        self.prev_dist_to_goal = np.linalg.norm(obs[6:9] - self.target_position)

        info ={}
        terminated = is_success
        truncated = self.current_step >= self.max_episode_steps

        # time_out = self.current_step >= self.max_episode_steps
        if terminated:
            info['is_success'] = True
        # 如果是因为超时而结束，info中就没有这个标志
        elif truncated:
            info['is_success'] = False
        # done = is_success or time_out

        return obs, reward, terminated,truncated, info

    def reset(self,seed=None):
        super().reset(seed=seed)

        # 2. 不要使用 p.resetSimulation()，只重置关节和位置
        # 重置机器人基座
        p.resetBasePositionAndOrientation(self.robotId[0], self.robotStartPos, self.robotStartOrientation)

        # 重置机器人关节 (这一步很重要，否则机器人会保持上一个episode的扭曲姿态)
        for i in range(p.getNumJoints(self.robotId[0])):
            p.resetJointState(self.robotId[0], i, targetValue=0, targetVelocity=0)

        # 随机生成 Cola 和 Table 的相对位置
        rand_x_1 = self.np_random.uniform(-0.7, 0.2)
        rand_x_2 = self.np_random.uniform(0.2, 0.7)

        # 重置 Cola 位置
        p.resetBasePositionAndOrientation(self.colaId, [rand_x_2, 0.8, 0], [0, 0, 0, 1])
        # 重置 Table 位置
        p.resetBasePositionAndOrientation(self.tableId, [rand_x_1, 0.8, 0], [0, 0, 0, 1])

        # 更新目标变量
        self.position = [rand_x_2, 0.8, 0]
        self.target_position = [rand_x_1, 0.8, 0.32]

        self.origin_distance = np.linalg.norm(np.array((self.position)) - np.array(self.target_position), axis=-1)
        p.changeDynamics(bodyUniqueId=self.colaId,
                         linkIndex=-1,  # 对 base 使用 -1
                         lateralFriction=25,
                         spinningFriction=5.0,
                         rollingFriction=0.001)
        obs = self.get_obs()
        cola_position = obs[6:9]
        self.prev_dist_to_goal = np.linalg.norm(cola_position - self.target_position)
        return obs,{}

    # 在 armEnv 类中


    def compute_reward(self, achieved_goal, goal, prev_dist):
        # 增加更多奖励成分
        SUCCESS_REWARD = 5
        POTENTIAL_SCALING = 1
        TIME_PENALTY = -0.05
        GRIP_REWARD = 0.5
        DISTANCE_PENALTY = 1

        gripper_state = p.getLinkState(self.robotId[0], self.end_effector_link_index)
        gripper_pos = np.array(gripper_state[4])
        dist_gripper_to_cola = np.linalg.norm(gripper_pos - achieved_goal)
        # 一个简单的负奖励，距离越近，惩罚越小（相当于奖励越大）
        distance_penalty = -DISTANCE_PENALTY*dist_gripper_to_cola

        current_dist = np.linalg.norm(achieved_goal - goal)
        is_success = (current_dist < self.distance_threshold)

        # 检查是否抓取到物体
        # gripper_state = p.getContactPoints(self.robotId[0], self.colaId)
        # has_contact = len(gripper_state) > 0
        contact_points_finger1 = p.getContactPoints(self.robotId[0], self.colaId,
                                                    linkIndexA=self.gripper_link_indices[0])
        contact_points_finger2 = p.getContactPoints(self.robotId[0], self.colaId,
                                                    linkIndexA=self.gripper_link_indices[1])

        has_contact = len(contact_points_finger1) > 0 and len(contact_points_finger2) > 0


        # 势能奖励
        potential_reward = max(POTENTIAL_SCALING * (prev_dist - current_dist),-5)

        # 抓取奖励
        # grip_reward = GRIP_REWARD if has_contact else 0
        grip_reward = GRIP_REWARD if has_contact else distance_penalty

        if is_success:
            total_reward = SUCCESS_REWARD
        else:
            total_reward = potential_reward + grip_reward + TIME_PENALTY



        return is_success, total_reward
    # def compute_reward(self, achieved_goal, goal,target_O):
    #     achieved_goal = np.array(achieved_goal)
    #     goal = np.array(goal)
    #     assert achieved_goal.shape == goal.shape
    #     dis =np.linalg.norm(achieved_goal - goal, axis=-1)
    #     flag = False
    #     # flag = dis<self.distance_threshold
    #     if dis<self.distance_threshold:
    #         if target_O ==[0,0,0]:
    #             flag = True
    #     if flag:
    #         reward = 2000
    #         # reward = 0
    #     else:
    #         reward = max(self.origin_distance - dis, 0)
    #         # reward = -1
    #     return [flag,reward]


    def get_obs(self):
        state = p.getLinkState(self.robotId[0], self.end_effector_link_index,computeLinkVelocity=1)
        pos = np.array(state[4])
        # orientation = np.array(p.getEulerFromQuaternion(state[5]))
        linear_v = np.array(state[6])
        # angular_v = np.array(state[7])
        cola_state = p.getBasePositionAndOrientation(self.colaId)
        cola_position = np.array(cola_state[0])
        # cola_orientation = p.getEulerFromQuaternion(np.array(cola_state[1]))
        cola_orientation = np.array(cola_state[1])
        angle1 = np.array([p.getJointState(self.robotId[0], 8)[0]])

        angle2 = np.array([p.getJointState(self.robotId[0], 10)[0]])
        goal_pos = self.target_position
        gripper_to_cola_vec = cola_position - pos
        cola_to_goal_vec = goal_pos - cola_position
        # obs = np.concatenate([pos, orientation,cola_position,cola_orientation,linear_v,angular_v,angle1,angle2])
        obs = np.concatenate([pos,linear_v,cola_position,cola_orientation,angle1,angle2,goal_pos,gripper_to_cola_vec,cola_to_goal_vec])
        return obs

    def take_act(self,Commands):
        def control_gripper(robot_id,rand1,rand2):
            # 将夹爪闭合到接近 0
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=8,
                controlMode=p.POSITION_CONTROL,
                targetPosition=-rand1,
                force=500
            )
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=11,
                controlMode=p.POSITION_CONTROL,
                targetPosition=rand1,
                force=500
            )
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=10,
                controlMode=p.POSITION_CONTROL,
                targetPosition=-rand2,
                force=500
            )
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=13,
                controlMode=p.POSITION_CONTROL,
                targetPosition=rand2,
                force=500
            )
        # def open_gripper(robot_id):
        #     # 将夹爪张开到一定角度
        #     p.setJointMotorControl2(
        #         bodyUniqueId=robot_id,
        #         jointIndex=8,  # 左爪根部
        #         controlMode=p.POSITION_CONTROL,
        #         targetPosition=-0.4,  # 角度(或弧度)，可自行调整
        #         force=50  # 能施加的力/扭矩上限
        #     )
        #     p.setJointMotorControl2(
        #         bodyUniqueId=robot_id,
        #         jointIndex=11,  # 右爪根部
        #         controlMode=p.POSITION_CONTROL,
        #         targetPosition=0.4,  # 往另一个方向打开
        #         force=50
        #     )
        state = p.getLinkState(self.robotId[0], self.end_effector_link_index)
        # pos = state[4]
        # new_pos = [pos[0]+Commands[0],pos[1]+Commands[1],pos[2]+Commands[2]]

        action_scale = 0.1  # 每步移动10cm
        pos = state[4]
        # 对动作进行缩放
        delta_pos = Commands[0:3] * action_scale
        new_pos = pos + delta_pos
        fixed_target_orientation = self.target_orientation_quat
        # orien = state[5]
        # current_ori = R.from_quat(orien)  # orien 是 [x, y, z, w]
        # delta_ori = R.from_euler('xyz', [Commands[3], Commands[4], Commands[5]])
        # target_ori = current_ori * delta_ori
        # new_orien = target_ori.as_quat()  # 这是 [x, y, z, w]
        targetPositionsJoints = p.calculateInverseKinematics(
            self.robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
            self.end_effector_link_index,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
            new_pos,  # 目标末端位置
            targetOrientation= fixed_target_orientation,
            # lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp
        )
        # if Commands[-1]>0.5:
        #     close_gripper(self.robotId[0])
        # if Commands[-1]<-0.5:
        #     open_gripper(self.robotId[0])
        control_gripper(self.robotId[0],Commands[-2],Commands[-1])
        p.setJointMotorControlArray(
            bodyIndex=self.robotId[0],  # 机械臂ID
            jointIndices=range(self.end_effector_link_index),  # 需要控制的关节索引
            controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
            targetPositions=targetPositionsJoints[0:self.end_effector_link_index],  # IK 求解得到的目标关节角
            forces=[500] * self.end_effector_link_index
            # targetVelocities = self.maxVelocities
        )

    def close(self):
        if self.physicsClient is not None:
            p.disconnect(self.physicsClient)
            self.physicsClient = None



