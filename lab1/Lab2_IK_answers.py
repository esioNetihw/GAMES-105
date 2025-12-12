import numpy as np
from scipy.spatial.transform import Rotation as R
def CCD_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    joint_parent = meta_data.joint_parent
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #特殊处理
    if len(path2) == 1 and path2[0] != 0:
        path2 = []
    #计算位姿
    orientation_chain = np.empty((len(path),), dtype=object)
    orientation_chain[0] = R.from_quat(joint_orientations[path[0]]) if len(path2) == 0 else R.from_quat(joint_orientations[path[1]])
    #计算rotation
    rotation_chain = np.empty((len(path),), dtype=object)
    rotation_chain[0] = R.from_quat([1, 0, 0, 0])
    #计算offset
    offset_chain = np.empty((len(path), 3))
    offset_chain[0] = np.array([0.0, 0.0, 0.0])
    position_chain = np.empty((len(path), 3))
    
    for i in range(1, len(path)):
        index = path[i]
        position_chain[i] = joint_positions[index]
        offset_chain[i] = meta_data.joint_initial_position[path[i]] - meta_data.joint_initial_position[path[i - 1]]
        if meta_data.joint_parent[i] == -1:
            joint_rotation = R.from_quat([1.,0.,0.,0]).as_quat()
        else:
            joint_rotation = (R.from_quat(joint_orientations[meta_data.joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
        if index in path2:
            # 旋转取反位姿需要用next节点不然乘顺序不对
            orientation_chain[i] = R.from_quat(joint_orientations[path[i + 1]])
            rotation_chain[i] = R.from_quat(joint_rotation).inv()
        else:
            orientation_chain[i] = R.from_quat(joint_orientations[index])
            rotation_chain[i] = R.from_quat(joint_rotation)

    iter_max_count = 10
    epsilon = 1e-3
    dampping = 1
    chain_length = len(path)
    for it in range(iter_max_count):
        if np.linalg.norm(position_chain[-1] - target_pose) < epsilon:
            break
        # 从末端节点往上遍历路径上的每个节点
        for i in range(len(path) - 2, -1, -1):
            current_idx = path[i]
            if meta_data.joint_parent[current_idx] == -1:
                continue
            if meta_data.joint_name[current_idx].endswith('_end'):
                continue
            vec_to_target = target_pose - position_chain[i]
            vec_to_end = position_chain[-1] - position_chain[i]
            rot_axis = np.cross(vec_to_end, vec_to_target)
            #单位化
            rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-8)
            dot_val = np.dot(vec_to_end, vec_to_target) / (np.linalg.norm(vec_to_end) * np.linalg.norm(vec_to_target) + 1e-8)
            #共线
            if np.linalg.norm(rot_axis) < 1e-6:
                continue
            rot_vec = R.from_rotvec(np.arccos(np.clip(dot_val, -1.0, 1.0)) * dampping * rot_axis)
            orientation_chain[i] = rot_vec * orientation_chain[i]
            rotation_chain[i] = orientation_chain[i - 1].inv() * orientation_chain[i]
            # FK更新链上的位置和朝向
            for j in range(i + 1, len(path)):
                orientation_chain[j] = orientation_chain[j - 1] * rotation_chain[j]
                position_chain[j] = position_chain[j - 1] + orientation_chain[j - 1].apply(offset_chain[j])
        distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))



    joint_rotations = np.empty(len(joint_orientations), dtype=object)
    for i in range(len(meta_data.joint_name)):
        if meta_data.joint_parent[i] == -1:
            joint_rotations[i] = R.from_quat([0.,0.,0.,1])
        else:
            joint_rotations[i] = R.from_quat(joint_orientations[meta_data.joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])
    #
    for i in range(len(path)):
        idx = path[i]
        joint_positions[idx] = position_chain[i]
        joint_rotations[idx] = rotation_chain[i].inv() if idx in path2 else rotation_chain[i]

    #
    if path2 == []:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * orientation_chain[0])

    #如果跨root单独处理
    if meta_data.joint_parent.index(-1) in path:
        root_index = path.index(meta_data.joint_parent.index(-1))
        if root_index != 0:
            joint_orientations[0] = orientation_chain[root_index].as_quat()
            joint_positions[0] = position_chain[root_index]

    # 最后计算一遍FK，得到更新后的position和orientation
    for idx in range(len(joint_positions)):
        if meta_data.joint_parent[idx] == -1:
            continue
        p_idx = meta_data.joint_parent[idx]
        joint_orientations[idx] = (R.from_quat(joint_orientations[p_idx]) * joint_rotations[idx]).as_quat()
        joint_offsets = meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[p_idx]
        joint_positions[idx] = joint_positions[p_idx] + R.from_quat(joint_orientations[p_idx]).apply(joint_offsets)
    return joint_positions, joint_orientations
def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    #CCD_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    #GroundTurth(meta_data, joint_positions, joint_orientations, target_pose)
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pose = np.array([relative_x, target_height, relative_z]) + joint_positions[0]
    joint_positions, joint_orientations = CCD_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations