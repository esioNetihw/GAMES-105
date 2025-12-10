import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = np.empty((0, 3))
    stack = []
    joint2_index = {}

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line == 'MOTION':
                break
            if line == '}':
                stack.pop()
            elif line == "{":
                stack.append(joint_name[-1])
            else:
                tokens = list(filter(None, line.split(" ")))
                current_join = stack[-1] if len(stack) > 0 else ""
                parent_idx = joint2_index.get(current_join, -1)
                if tokens[0] == "ROOT" or tokens[0] == "JOINT":
                    joint_name.append(tokens[1])
                    joint_parent.append(parent_idx)
                    joint2_index[tokens[1]] = len(joint_name) - 1
                elif tokens[0] == "End":
                    joint_name.append(current_join + "_end")
                    joint_parent.append(joint2_index[current_join])
                    joint2_index[current_join + "_end"] = len(joint_name) - 1
                elif tokens[0] == "OFFSET":
                    joint_offset = np.append(joint_offset, np.array([[float(tokens[1]), float(tokens[2]), float(tokens[3])]]), axis=0)
    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset

def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    joint_positions = np.empty((0, 3))
    joint_orientations = np.empty((0, 4))

    frame = motion_data[frame_id]
    #root数据
    joint_positions = np.append(joint_positions, np.array([frame[0:3]]), axis=0)
    joint_orientations = np.append(joint_orientations, np.array([R.from_euler('XYZ', frame[3:6], degrees=True).as_quat()]), axis=0)
    real_idx = 1
    for idx in range(1, len(joint_name)):
        name = joint_name[idx]
        p_idx = joint_parent[idx]
        if name.endswith("_end"):
            local_rotation = R.from_euler('XYZ', [0, 0, 0], degrees=True)
        else:
            local_rotation = R.from_euler('XYZ', frame[real_idx * 3 + 3 : real_idx * 3 + 6], degrees=True)
            real_idx = real_idx + 1
        parent_roation = R.from_quat(joint_orientations[p_idx])
        joint_orientations = np.append(joint_orientations, np.array([(parent_roation * local_rotation).as_quat()]), axis=0)
        joint_positions = np.append(joint_positions, np.array([joint_positions[p_idx] + parent_roation.apply(joint_offset[idx])]), axis=0)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    
    Apose_joint_name, Apose_joint_parent, Apose_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    Tpose_joint_name, Tpose_joint_parent, Tpose_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)

    #提取Apose
    Apose_motion_data = load_motion_data(A_pose_bvh_path)
    motion_data = np.empty((Apose_motion_data.shape[0], Apose_motion_data.shape[1]))

    #计算各个关节的全局朝向的偏移
    orientation_A2T = {}
    Apose_position, Apose_orientations = part2_forward_kinematics(Apose_joint_name, Apose_joint_parent, Apose_joint_offset, Apose_motion_data, 0)
    for i in range(0, len(Apose_orientations)):
        name = Apose_joint_name[i]
        p_name = Apose_joint_name[Apose_joint_parent[i]]
        if name == "lShoulder" or name == "rShoulder" or p_name in orientation_A2T:
            orientation_A2T[name] = R.from_quat(Apose_orientations[i])
    Apose_name_to_idx = {}
    joint_idx = 0
    for j in range(0, len(Apose_joint_name)):
        name = Apose_joint_name[j]
        if name.endswith("_end"):
            continue
        Apose_name_to_idx[name] = joint_idx
        joint_idx = joint_idx + 1
    Tpose_name_to_Apose_idx = {}
    for j in range(0, len(Tpose_joint_name)):
        name = Tpose_joint_name[j]
        if name.endswith("_end"):
            continue
        Tpose_name_to_Apose_idx[name] = Apose_name_to_idx[name]

    for i in range(0, Apose_motion_data.shape[0]):
        frame = Apose_motion_data[i]
        new_frame = motion_data[i]
        #root位移
        new_frame[0:3] = frame[0:3]
        #填充Tpose的各个关节的旋转
        joint_idx = 0
        for j in range(0, len(Tpose_joint_name)):
            name = Tpose_joint_name[j]
            p_name = Tpose_joint_name[Tpose_joint_parent[j]]
            if name.endswith("_end"):
                continue
            Apose_idx = Tpose_name_to_Apose_idx[name]
            if p_name in orientation_A2T or name in orientation_A2T:
                Ri = R.from_euler("XYZ", frame[Apose_idx * 3 + 3 : Apose_idx * 3 + 6], degrees=True)
                #print(Ri.as_euler("XYZ", degrees=True))
                parent_orientation_diff = orientation_A2T.get(p_name, R.from_euler('XYZ', [0, 0, 0], degrees=True))
                #print(Ri.as_euler("XYZ", degrees=True))
                orientation_diff_inv = orientation_A2T.get(name, R.from_euler('XYZ', [0, 0, 0], degrees=True)).inv()
                new_frame[joint_idx * 3 + 3 : joint_idx * 3 + 6] = (parent_orientation_diff * Ri * orientation_diff_inv).as_euler("XYZ", degrees=True)
            else:
                new_frame[joint_idx * 3 + 3 : joint_idx * 3 + 6] = frame[Apose_idx * 3 + 3 : Apose_idx * 3 + 6]
            joint_idx = joint_idx + 1
        #break
    return motion_data
