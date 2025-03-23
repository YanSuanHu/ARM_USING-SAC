# import pybullet as p
# import pybullet_data #pybullet自带的一些模型
# import os
# p.connect(p.GUI) #连接到仿真环境，p.DIREACT是不显示仿真界面,p.GUI则为显示仿真界面
# p.setGravity(0,0,-10) #设定重力
# p.resetSimulation() #重置仿真环境
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
# planeId = p.loadURDF("plane.urdf") #加载外部平台模型
# p.setAdditionalSearchPath(os.path.abspath('models'))
# objects = p.loadSDF('kuka_with_gripper.sdf') #加载机械臂，flags=9代表取消自碰撞，详细教程可以参考pybullet的官方说明文档

import pybullet as p
import time
import pybullet_data  # pybullet 自带的一些模型
import os
import numpy as np
from pybullet import stepSimulation


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
        targetPosition=-0.5,  # 角度(或弧度)，可自行调整
    )
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=11,  # 右爪根部
        controlMode=p.POSITION_CONTROL,
        targetPosition=0.5,  # 往另一个方向打开
    )
# 连接到仿真环境，p.DIRECT 不显示仿真界面，p.GUI 则显示仿真界面
physicsClient = p.connect(p.GUI)  # 或者 p.DIRECT
# 再添加自定义模型所在的目录（确保 'models' 路径正确）
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载平面
planeId = p.loadURDF("plane.urdf")
p.setAdditionalSearchPath(os.path.abspath("../models"))

p.setGravity(0, 0, -10)  # 设置重力


# 设置机械臂初始位姿
robotStartPos = [0, 0, 0]
robotStartOrientation = p.getQuaternionFromEuler([0,0, 0])


# 加载机械臂（SDF 格式）
# robotId = p.loadURDF() "" # 加载机械臂
robotId = p.loadSDF("kuka_with_gripper.SDF")
p.resetBasePositionAndOrientation(robotId[0], robotStartPos, robotStartOrientation)
                    # basePosition=robotStartPos, baseOrientation=robotStartOrientation,
                    #   useFixedBase=True)
# tableId = p.loadURDF("table_square/table_square.urdf",[0.6,0.6,0],globalScaling=0.5)
culaId = p.loadURDF("Household-items-urdfs/urdf/cola.urdf",[0,1,0],p.getQuaternionFromEuler([0,0,0]))
# while True:
#     stepSimulation()
# p.resetBasePositionAndOrientation(robotId, robotStartPos, robotStartOrientation)
# 添加 pybullet 自带的模型搜索路径

end_effector_link_index = 11  # 根据模型调整索引，通常是第7个链接（索引6）
# for i in range(11):
#     initial_state = p.getLinkState(robotId, i)
#     print("在这里！",initial_state[0])
# startPos = initial_state[0]  # 初始末端位置
p.changeDynamics(bodyUniqueId=robotId[0],
                 linkIndex=7,
                 lateralFriction=5.0,
                 spinningFriction=1.0,
                 rollingFriction=0.001)
p.changeDynamics(bodyUniqueId=culaId,
                 linkIndex=-1,  # 对 base 使用 -1
                 lateralFriction=5.0,
                 spinningFriction=1.0,
                 rollingFriction=0.001)

num_joints = p.getNumJoints(robotId[0])
joint_info = [p.getJointInfo(robotId[0], i) for i in range(num_joints)]
# print("joint_info: ", joint_info)
state = p.getLinkState(robotId[0], 7)
print("这里！",state)
# while True:
#     p.stepSimulation()
#     time.sleep(1/50)
#
# # 加载一个立方体（此处命名为 cylinderId 仅作示例）
# # cubeStartPos = [0, 0, 0]
# # cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# # cubeId = p.loadURDF("cube.urdf", cubeStartPos, cubeStartOrientation)
#
p.setTimeStep(1.0 / 240)
p.setRealTimeSimulation(0)


# targetPositionsJoints = p.calculateInverseKinematics(
#         robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
#         7,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
#         [0.6,0.6,0.32],  # 目标末端位置
#     )
# p.setJointMotorControlArray(
#         bodyUniqueId=robotId[0],  # 机械臂ID
#         jointIndices=range(7),  # 需要控制的关节索引
#         controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
#         targetPositions=targetPositionsJoints[0:7]  # IK 求解得到的目标关节角
# )
# p.stepSimulation()
# time.sleep(0.5)
# close_gripper(robotId[0])
# p.stepSimulation()
# time.sleep(0.5)
# targetPositionsJoints = p.calculateInverseKinematics(
#         robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
#         7,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
#         [0,-1,0],  # 目标末端位置
#     )
# p.setJointMotorControlArray(
#         bodyUniqueId=robotId[0],  # 机械臂ID
#         jointIndices=range(7),  # 需要控制的关节索引
#         controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
#         targetPositions=targetPositionsJoints[0:7]  # IK 求解得到的目标关节角
# )
# while True:
#     time.sleep(2)
#     open_gripper(robotId[0])
#     for j in range(200):
#         p.stepSimulation()
#         time.sleep(0.01)
#     close_gripper(robotId[0])
#     for j in range(200):
#         p.stepSimulation()
#         time.sleep(0.01)
#     open_gripper(robotId[0])
#     for j in range(200):
#         p.stepSimulation()
#         time.sleep(0.01)
#     time.sleep(2)
#     close_gripper(robotId[0])
#     for j in range(200):
#         p.stepSimulation()
#         time.sleep(0.01)
#     open_gripper(robotId[0])
#     for j in range(200):
#         p.stepSimulation()
#         time.sleep(0.01)



# 获取机械臂的关节数

#print("num_joints: ", num_joints)

# 获取每个关节的信息


# 设置插值步数与目标位置
stepNum = 500
endPos = [0, 0.85, 0.6]

end_effector_link_index = 7
initial_state = p.getLinkState(robotId[0], end_effector_link_index)
startPos = initial_state[0]  # 初始末端位置

# 将起始与目标位置转换为 numpy 数组
startPos_array = np.array(startPos)

endPos_array = np.array(endPos)
# print("看我！",p.getBasePositionAndOrientation(culaId))
# 线性插值步进
step_array = (endPos_array - startPos_array) / stepNum
# targetOrientation = p.getQuaternionFromEuler([0, np.pi/2, 0])  # 示例姿态
# 演示循环
open_gripper(robotId[0])
for j in range(50):
    p.stepSimulation()  # 推进仿真一步
    time.sleep(0.01)
for j in range(stepNum+1):
    # 1. 计算本次迭代要移动到的末端位置
    alpha = float(j) / stepNum
    robotStepPos = startPos_array + alpha * (endPos_array - startPos_array)


    # 2. 利用逆运动学 (IK) 求解出末端到达 robotStepPos 时，机械臂各关节的目标角度

    targetPositionsJoints = p.calculateInverseKinematics(
        robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
        7,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
        robotStepPos,  # 目标末端位置
        targetOrientation= p.getQuaternionFromEuler([-np.pi/2,0, 0]),
    )

    # 3. 设置机械臂各关节的控制模式和目标关节角
    p.setJointMotorControlArray(
        bodyUniqueId=robotId[0],  # 机械臂ID
        jointIndices=range(7),  # 需要控制的关节索引
        controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
        targetPositions=targetPositionsJoints[:7]  # IK 求解得到的目标关节角

    )
    p.stepSimulation()  # 推进仿真一步
    time.sleep(0.01)  # 等待 0.1 秒（放慢可视化速度）

    # 5. 更新起始位置为本次迭代末端位置，以便下一步可以在此基础上继续
    if j %100==0:
        print(robotStepPos)
endPos = [0, 0.85, 0.1]
initial_state = p.getLinkState(robotId[0], end_effector_link_index)
startPos = initial_state[0]  # 初始末端位置
startPos_array = np.array(startPos)
endPos_array =np.array(endPos)

step_array = (endPos_array - startPos_array) / stepNum
for j in range(stepNum+1):
    # 1. 计算本次迭代要移动到的末端位置
    alpha = float(j) / stepNum
    robotStepPos = startPos_array + alpha * (endPos_array - startPos_array)


    # 2. 利用逆运动学 (IK) 求解出末端到达 robotStepPos 时，机械臂各关节的目标角度
    targetPositionsJoints = p.calculateInverseKinematics(
        robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
        7,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
        robotStepPos,  # 目标末端位置
        targetOrientation=p.getQuaternionFromEuler([-np.pi/2,0, 0]),
    )

    # 3. 设置机械臂各关节的控制模式和目标关节角
    p.setJointMotorControlArray(
        bodyUniqueId=robotId[0],  # 机械臂ID
        jointIndices=range(7),  # 需要控制的关节索引
        controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
        targetPositions=targetPositionsJoints[:7]  # IK 求解得到的目标关节角

    )
    p.stepSimulation()  # 推进仿真一步
    time.sleep(0.01)  # 等待 0.1 秒（放慢可视化速度）

    # 5. 更新起始位置为本次迭代末端位置，以便下一步可以在此基础上继续
    if j %100==0:
        print(robotStepPos)
close_gripper(robotId[0])
for j in range(50):
    p.stepSimulation()  # 推进仿真一步
    time.sleep(0.01)

endPos = [0, 0.85, 0.4]
initial_state = p.getLinkState(robotId[0], end_effector_link_index)
startPos = initial_state[0]  # 初始末端位置
startPos_array = np.array(startPos)
endPos_array =np.array(endPos)

step_array = (endPos_array - startPos_array) / stepNum
for j in range(stepNum+1):
    # 1. 计算本次迭代要移动到的末端位置
    alpha = float(j) / stepNum
    robotStepPos = startPos_array + alpha * (endPos_array - startPos_array)


    # 2. 利用逆运动学 (IK) 求解出末端到达 robotStepPos 时，机械臂各关节的目标角度
    targetPositionsJoints = p.calculateInverseKinematics(
        robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
        7,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
        robotStepPos,  # 目标末端位置
        targetOrientation=None,
    )

    # 3. 设置机械臂各关节的控制模式和目标关节角
    p.setJointMotorControlArray(
        bodyUniqueId=robotId[0],  # 机械臂ID
        jointIndices=range(7),  # 需要控制的关节索引
        controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
        targetPositions=targetPositionsJoints[:7]  # IK 求解得到的目标关节角

    )
    p.stepSimulation()  # 推进仿真一步
    time.sleep(0.01)  # 等待 0.1 秒（放慢可视化速度）

    # 5. 更新起始位置为本次迭代末端位置，以便下一步可以在此基础上继续
    if j %100==0:
        print(robotStepPos)

endPos = [0, -1, 0.1]
initial_state = p.getLinkState(robotId[0], end_effector_link_index)
startPos = initial_state[0]  # 初始末端位置
startPos_array = np.array(startPos)
endPos_array =np.array(endPos)

step_array = (endPos_array - startPos_array) / stepNum
for j in range(stepNum+1):
    # 1. 计算本次迭代要移动到的末端位置
    alpha = float(j) / stepNum
    robotStepPos = startPos_array + alpha * (endPos_array - startPos_array)


    # 2. 利用逆运动学 (IK) 求解出末端到达 robotStepPos 时，机械臂各关节的目标角度
    targetPositionsJoints = p.calculateInverseKinematics(
        robotId[0],  # 机械臂的ID（如果loadSDF返回列表,一般取robotId[0]）
        7,  # 通常是末端执行器的链接索引(取决于具体机械臂模型)
        robotStepPos,  # 目标末端位置
        targetOrientation=None,
    )

    # 3. 设置机械臂各关节的控制模式和目标关节角
    p.setJointMotorControlArray(
        bodyUniqueId=robotId[0],  # 机械臂ID
        jointIndices=range(7),  # 需要控制的关节索引
        controlMode=p.POSITION_CONTROL,  # 控制模式：位置控制
        targetPositions=targetPositionsJoints[:7]  # IK 求解得到的目标关节角

    )
    p.stepSimulation()  # 推进仿真一步
    time.sleep(0.01)  # 等待 0.1 秒（放慢可视化速度）

    # 5. 更新起始位置为本次迭代末端位置，以便下一步可以在此基础上继续
    if j %100==0:
        print(robotStepPos)