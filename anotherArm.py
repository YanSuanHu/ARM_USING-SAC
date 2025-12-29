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
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1 / 240)  # 标准仿真步长

        # 2. 定义动作空间 (Continuous Control)
        # [dx, dy, dz, gripper_ctrl]
        # dx, dy, dz: 末端执行器的位移增量 (-1 到 1，会被缩放)
        # gripper_ctrl: 夹爪开合控制 (-1: 开, 1: 关)
        self.action_dim = 4
        low = np.array([-1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 动作缩放因子 (每一步最大移动 5cm)
        self.action_scale = 0.05

        # 3. 定义观测空间
        # 包括: EE_pos(3), EE_vel(3), Cube_pos(3), Rel_pos(3), Gripper_width(1) = 13维
        self.observation_dim = 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # 加载静态物体
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

 
        # 设置视角
        if self.render_mode:
            self.set_gui_view()

        # 加载机器人
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # 初始化变量
        self.cube_id = None
        self.max_steps = 200  # 增加步数，因为是连续控制
        self.current_step = 0
        self.gripper_range = [0, 0.085]  # [closed, open] 注意：robotiq通常0是关，0.085是开，或者反过来，视模型而定

        # 修改摩擦力
        for link_id in [12, 17]:
            p.changeDynamics(self.robot.id, link_id,
                             lateralFriction=100.0,
                             spinningFriction=1.0,
                             frictionAnchor=1)

        # 工作空间限制 (防止机械臂乱跑)
        self.workspace_low = np.array([0.2, -0.5, 0.63])  # 根据桌子高度调整
        self.workspace_high = np.array([0.8, 0.5, 1.2])
        self.previous_cube_height = 0.65

    def set_gui_view(self):
        camera_distance = 1.2
        camera_yaw = 90
        camera_pitch = -30
        camera_target = [0.5, 0, 0.6]
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 1. 重置机器人姿态 (保持原逻辑，不添加随机噪声)
        self.robot.reset_pose()

        # 2. 重置方块位置
        x_range = np.arange(0.4, 0.7, 0.1)
        y_range = np.arange(-0.2, 0.2, 0.1)

        cube_start_pos = [
            np.random.choice(x_range) + np.random.uniform(-0.05, 0.05),  # 加一点微小连续噪声
            np.random.choice(y_range) + np.random.uniform(-0.05, 0.05),
            0.65  # 略高于桌面，防止嵌入
        ]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])

        if self.cube_id is not None:
            p.removeBody(self.cube_id)

        self.cube_id = p.loadURDF("./models/urdf/cube_blue.urdf", cube_start_pos, cube_start_orn, globalScaling=0.8)
        # 增加方块摩擦力以便抓取
        p.changeDynamics(self.cube_id, -1, lateralFriction=2.0, spinningFriction=0.01, rollingFriction=0.001)

        # 3. 预热几步
        for _ in range(10):
            p.stepSimulation()

        self.target_pos = np.array(cube_start_pos)
        self.previous_cube_height = 0.65

        # 获取初始观测
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # 1. 解析动作
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_pos_total = action[:3] * self.action_scale  # 总位移，比如 0.05m

        # 获取当前状态
        current_ee_state = p.getLinkState(self.robot.id, self.robot.eef_id)
        current_pos = np.array(current_ee_state[0])

        # 计算最终目标
        target_pos_final = current_pos + delta_pos_total
        target_pos = np.clip(target_pos_final, self.workspace_low, self.workspace_high)

        # 【关键修改】 姿态固定 (垂直向下)
        target_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])

        # 2. 夹爪控制 (平滑处理)
        # 我们假设夹爪动作不需要插值，直接到位即可，或者也在这里分步
        gripper_width = 0.085 * (1 - (action[3] + 1) / 2)
        self.robot.move_arm_ik(target_pos, target_orn)

        # 3. 夹爪控制
        # action[3] -> [-1, 1] 映射到 [0.085, 0]
        # 注意：这里可能需要根据你的模型反转，如果动作不对，把 (1 - ...) 改为 (1 + ...)
        gripper_width = 0.085 * (1 - (action[3] + 1) / 2)
        self.robot.move_gripper(gripper_width)

        # 4. 步进仿真
        for _ in range(20):
            p.stepSimulation()

        # 4. 获取观测和奖励 (只在动作做完后计算一次)
        obs = self._get_obs()
        reward, is_success = self._compute_reward(obs)

        terminated = is_success
        truncated = self.current_step >= self.max_steps
        info = {'is_success': is_success}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 机械臂末端状态
        ee_state = p.getLinkState(self.robot.id, self.robot.eef_id, computeLinkVelocity=1)
        ee_pos = np.array(ee_state[0])
        ee_vel = np.array(ee_state[6])

        # 物体状态
        cube_state = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_state[0])

        # 相对位置
        rel_pos = cube_pos - ee_pos

        # 夹爪宽度 (简化，直接用mimic joint的状态)
        # 假设 joint 12 是左指
        gripper_state = p.getJointState(self.robot.id, self.robot.mimic_parent_id)
        gripper_angle = np.array([gripper_state[0]])

        # 拼接向量
        obs = np.concatenate([ee_pos, ee_vel, cube_pos, rel_pos, gripper_angle])
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
        self.eef_id = 7  # 末端执行器 Link ID
        self.arm_num_dofs = 6
        # 初始姿态 (Rest Pose)
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.id = None
        self.max_velocity = 2.0  # 限制一下最大速度，让动作更平滑

    def load(self):
        # flags = p.URDF_USE_SELF_COLLISION
        self.id = p.loadURDF('./models/urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()
        self.reset_pose()

    # def reset_pose(self):
    #     # 强制重置关节角度
    #     for i, joint_id in enumerate(self.arm_controllable_joints):
    #         p.resetJointState(self.id, joint_id, self.arm_rest_poses[i])
    #     # 重置夹爪为张开状态
    #     self.move_gripper(0.085)

    def reset_pose(self):
        # 1. 瞬移：强制将关节状态重置为 rest_pose (视觉上瞬间到位)
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, self.arm_rest_poses[i])

        # 2. 【新增】保持：告诉电机“你的目标就是待在这里”，防止它往 0 度跑
        # 这一步非常重要，否则一运行 stepSimulation，机械臂就会弹回 0 度
        p.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.arm_controllable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.arm_rest_poses,  # 目标设为初始姿态
            forces=[100.0] * self.arm_num_dofs  # 给予足够的力矩保持姿态
        )

        # 重置夹爪为张开状态
        self.move_gripper(0.085)

    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity',
                                'controllable'])
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                          jointMaxVelocity, controllable)
            )

        # 前6个是机械臂关节
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        # Robotiq 85 夹爪的连杆约束设置
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        # 找到父关节 ID
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]

        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if
                                       joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            p.setJointMotorControl2(self.id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
  
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=0.8)


    # def move_gripper(self, open_length):
    #     """
    #     控制夹爪开合
    #     open_length: 目标宽度 (0 ~ 0.085m)
    #     """
    #     # 限制范围
    #     open_length = np.clip(open_length, 0, 0.085)

    #     # 根据几何关系计算角度 (Robotiq 85 specific)
    #     open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)

    #     # 只控制 mimic parent 关节，其他关节会通过 Constraint 自动跟随
    #     p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
    #                             force=100, maxVelocity=2.0)
        
    def move_gripper(self, open_length):
        open_length = np.clip(open_length, 0, 0.085)

        # 线性映射：open_length=0.085 -> angle=0 (张开)
        #         open_length=0     -> angle=0.8 (闭合)
        target = (1.0 - open_length / 0.085) * 0.8

        # 再保险：按 joint limit 裁剪
        jinfo = p.getJointInfo(self.id, self.mimic_parent_id)
        lo, hi = jinfo[8], jinfo[9]
        target = float(np.clip(target, lo, hi))

        p.setJointMotorControl2(
            self.id, self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=200,        # 给足力
            maxVelocity=5.0   # 快一点，方便观察
        )


    def move_arm_ik(self, target_pos, target_orn):
        """
        使用逆运动学控制机械臂移动到目标位置
        """
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id, target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
            maxNumIterations=20  # 减少迭代次数提高速度
        )

        # 设置关节位置控制
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    maxVelocity=self.max_velocity)

    def get_current_ee_position(self):
        return p.getLinkState(self.id, self.eef_id)






import time
import math



def test_gripper_visualization():
    """
    一个专门用于可视化和测试 Robotiq 85 夹爪开合功能的脚本。
    它会创建一个滑动条，让你可以手动控制夹爪的目标开合宽度。
    """
    # 1. 初始化 PyBullet GUI
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0) # 我们将手动步进

    # 2. 加载基础场景
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

    # 3. 加载机器人
    # 将机器人放置在桌子上方，方便观察
    robot = UR5Robotiq85(pos=[0.5, 0, 0.63], ori=[0, 0, 0])
    robot.load()

    # 4. 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=90,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0, 0.7]
    )

    # 5. 创建一个 GUI 滑动条来控制夹爪
    # open_length 范围是 0 (闭合) 到 0.085 (完全张开)
    gripper_slider = p.addUserDebugParameter("Gripper Opening", 0, 0.085, 0.085)

    print("="*50)
    print("夹爪可视化测试已启动。")
    print("请在GUI窗口中拖动 'Gripper Opening' 滑动条来控制夹爪。")
    print("按 Ctrl+C 或关闭窗口来退出。")
    print("="*50)

    # 6. 仿真循环
    try:
        while True:
            # 读取滑动条的值
            target_opening_length = p.readUserDebugParameter(gripper_slider)

            # 调用 move_gripper 函数
            robot.move_gripper(target_opening_length)

            # 步进仿真
            p.stepSimulation()
            
            # 稍微延时，让渲染更平滑
            time.sleep(1./240.)

    except KeyboardInterrupt:
        print("\n程序已退出。")
    finally:
        p.disconnect()
  


if __name__ == "__main__":
    print('start')
    test_gripper_visualization()