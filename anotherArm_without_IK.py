import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import namedtuple
import math
import random

class UR5RobotiqEnv(gym.Env):
    def __init__(self, visuable=False):
        super(UR5RobotiqEnv, self).__init__()

        # 1. 初始化 PyBullet
        self.render_mode = visuable
        self.previous_cube_height = 0.65
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1 / 240)

        # 2. 定义动作空间 (关节控制)
        # 6个手臂关节 + 1个夹爪关节
        self.action_dim = 7
        # 每个动作的范围都是[-1, 1]，代表关节移动的增量比例
        low = -1 * np.ones(self.action_dim, dtype=np.float32)
        high = 1 * np.ones(self.action_dim, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 动作缩放因子 (每一步关节最大移动角度)
        self.joint_action_scale = np.deg2rad(10) # 每步最多移动5度

        # 3. 定义观测空间
        # 原有: EE_pos(3), EE_vel(3), Cube_pos(3), Rel_pos(3), Gripper_angle(1) = 13维
        # 新增: Joint_pos(6), Joint_vel(6)
        # 总维度: 13 + 6 + 6 = 25维
        self.observation_dim = 25
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # 加载静态物体
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

        if self.render_mode:
            self.set_gui_view()

        # 加载机器人
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # 初始化变量
        self.cube_id = None
        self.max_steps = 200
        self.current_step = 0

        # 增加夹爪摩擦力
        for link_id in [12, 17]: # finger link ids
            p.changeDynamics(self.robot.id, link_id, lateralFriction=5.0)

    def set_gui_view(self):
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.7])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_cube_height = 0.65
        self.robot.reset_pose()

        # 重置方块位置
        x_pos = random.uniform(0.4, 0.6)
        y_pos = random.uniform(-0.2, 0.2)
        cube_start_pos = [x_pos, y_pos, 0.65]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])

        if self.cube_id is not None:
            p.removeBody(self.cube_id)
        self.cube_id = p.loadURDF("./models/urdf/cube_blue.urdf", cube_start_pos, cube_start_orn, globalScaling=0.8)
        p.changeDynamics(self.cube_id, -1, lateralFriction=2.0)

        for _ in range(10):
            p.stepSimulation()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # --- 核心改动：关节控制 ---
        # 1. 解析手臂关节动作
        arm_action = action[:6]
        current_joint_positions = self.robot.get_arm_joint_positions()
        # 计算目标关节角度 = 当前角度 + 动作增量
        target_joint_positions = current_joint_positions + arm_action * self.joint_action_scale
        
        # 将目标角度发送给电机
        p.setJointMotorControlArray(
            bodyIndex=self.robot.id,
            jointIndices=self.robot.arm_controllable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[self.robot.joints[i].maxForce for i in self.robot.arm_controllable_joints],
            positionGains=[0.2] * len(self.robot.arm_controllable_joints), # 使用默认的 gain 值
            velocityGains=[1] * len(self.robot.arm_controllable_joints)
        )

        # 2. 解析并执行夹爪动作
        gripper_action = action[6]
        # 将[-1, 1]映射到[0, 0.085] (0:闭合, 0.085:张开)
        gripper_opening_length = (gripper_action + 1) / 2 * 0.085
        self.robot.move_gripper(gripper_opening_length)

        # 3. 步进仿真
        for _ in range(20):
            p.stepSimulation()

        # 4. 获取观测和奖励
        obs = self._get_obs()
        reward, is_success = self._compute_reward(obs)

        terminated = is_success
        truncated = self.current_step >= self.max_steps
        info = {'is_success': is_success}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 原有观测
        ee_state = p.getLinkState(self.robot.id, self.robot.eef_id, computeLinkVelocity=1)
        ee_pos = np.array(ee_state[0])
        ee_vel = np.array(ee_state[6])
        cube_state = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_state[0])
        rel_pos = cube_pos - ee_pos
        gripper_state = p.getJointState(self.robot.id, self.robot.mimic_parent_id)
        gripper_angle = np.array([gripper_state[0]])

        # --- 新增观测：关节状态 ---
        joint_states = p.getJointStates(self.robot.id, self.robot.arm_controllable_joints)
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])

        # 拼接所有观测向量
        obs = np.concatenate([
            ee_pos, ee_vel, cube_pos, rel_pos, gripper_angle,
            joint_positions, joint_velocities
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, obs):
        ee_pos = obs[:3]
        cube_pos = obs[6:9]

        # 距离
        dist = np.linalg.norm(ee_pos - cube_pos)

        # 1. 距离惩罚 (希望距离越小越好)
        reward = -dist

        # 2. 接触奖励
        contact_points = p.getContactPoints(self.robot.id, self.cube_id)
        # The link index for the robot is the 3rd element in the contact point tuple
        contact_links = set(item[3] for item in contact_points)
        
        # Gripper finger link IDs, assuming they are 12 and 17 from your friction setup
        left_finger_contact = 12 in contact_links
        right_finger_contact = 17 in contact_links

        grasp_reward = 0
        if left_finger_contact and right_finger_contact:
            # Strong reward for a proper two-fingered grasp
            grasp_reward = 1.0
        elif left_finger_contact or right_finger_contact:
            # Small reward for one-fingered contact to guide the agent,
            # but not enough to make it a stable strategy.
            grasp_reward = 0.25
        reward+=grasp_reward

        current_cube_height = cube_pos[2]
        height_difference = current_cube_height - self.previous_cube_height
        
        # 给予一个与高度变化量成正比的奖励
        # 向上移动会获得正奖励，向下移动会获得负奖励（惩罚）
        # 乘以一个较大的系数来放大这个信号
        lift_reward = height_difference * 100
        reward += lift_reward
        self.previous_cube_height = current_cube_height

        # 3. 成功判定 (抬起物体)
        # 假设桌面高度是 ~0.63，如果物体高度超过 0.75 认为抬起成功
        is_success = False
        if cube_pos[2] > 0.75 and left_finger_contact and right_finger_contact:
            reward += 200.0
            is_success = True
            # print("Success Picked!")

        return reward, is_success

    def close(self):
        p.disconnect()


class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.id = None

    def load(self):
        self.id = p.loadURDF('./models/urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()
        self.reset_pose()

    def reset_pose(self):
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, self.arm_rest_poses[i])
        
        p.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.arm_controllable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.arm_rest_poses,
            forces=[100.0] * self.arm_num_dofs
        )
        self.move_gripper(0.085) # 张开夹爪

    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo', ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity = info[0], info[1].decode("utf-8"), info[2], info[8], info[9], info[10], info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable))
        
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1, 'left_inner_knuckle_joint': 1, 'right_inner_knuckle_joint': 1, 'left_inner_finger_joint': -1, 'right_inner_finger_joint': -1}
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id, jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_length):
        open_length = np.clip(open_length, 0, 0.085)
        # 线性映射: open_length=0.085 -> angle=0 (张开), open_length=0 -> angle=0.8 (闭合)
        target_angle = (1.0 - open_length / 0.085) * 0.8
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=target_angle, force=200)

    def get_arm_joint_positions(self):
        """获取当前手臂6个关节的角度"""
        joint_states = p.getJointStates(self.id, self.arm_controllable_joints)
        return np.array([state[0] for state in joint_states])

if __name__ == "__main__":
    # 一个简单的测试函数，用于验证环境是否能正常运行
    env = UR5RobotiqEnv(visuable=True)
    obs, _ = env.reset()
    print("Observation space shape:", env.observation_space.shape)
    print("Action space shape:", env.action_space.shape)
    
    for i in range(1000):
        # 执行一个随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if (i+1) % 200 == 0:
            print(f"Step {i+1}, Reward: {reward}, Success: {info.get('is_success')}")
            obs, _ = env.reset()

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()