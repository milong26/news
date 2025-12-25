"""
加载数据集
用T_cam_base计算新的action
需要计算新的state吗？？？
需要增量吗？
"""
import torch
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================
# 用户配置
# ============================================================
DATASETS = [
    "collect_data/first_camera",
    "collect_data/second_camera",
    "collect_data/third_camera"
]

# 每个数据集对应的 T_cam_base
T_CAM_BASE_LIST = [
    np.array([
        [-0.63790803, -0.1445514,  -0.53461799, 0.1674904 ],
        [ 0.22263082,  0.80519111, -0.4422212,  0.06543871],
        [ 0.50083595, -0.55931218, -0.48165269, 0.25525138],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [-0.5764445,  -0.15825889, -0.58744738, 0.12830973],
        [ 0.25145675, 0.81434572, -0.41448413, 0.05341024],
        [ 0.54732247,-0.54920714, -0.42757153, 0.24731953],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [-0.44517692, -0.15646792, -0.72388766, 0.14254446],
        [ 0.32365284, 0.82640186, -0.35126395,-0.00328258],
        [ 0.66044639,-0.52476247, -0.31207214, 0.31977424],
        [0., 0., 0., 1.]
    ])
]

# ============================================================
# 坐标变换函数
# ============================================================
def transform_pose_base_to_cam(pos_base, rotvec_base, gripper, T_cam_base):
    """
    将base坐标系下的绝对位姿 + gripper 转到camera坐标系
    pos_base: [x, y, z]
    rotvec_base: [wx, wy, wz] 旋转向量
    gripper: float
    T_cam_base: 4x4 numpy array
    """
    R_cam_base = T_cam_base[:3, :3]
    t_cam_base = T_cam_base[:3, 3]

    # 位置
    pos_cam = R_cam_base @ pos_base + t_cam_base

    # 旋转
    R_base = R.from_rotvec(rotvec_base).as_matrix()
    R_cam = R_cam_base @ R_base
    rotvec_cam = R.from_matrix(R_cam).as_rotvec()

    return np.concatenate([pos_cam, rotvec_cam, [gripper]])


# ============================================================
# 主处理函数
# ============================================================
def process_dataset(repo_id, T_cam_base):
    print(f"\n📁 处理数据集: {repo_id}")
    
    # 加载旧数据集
    old_dataset = LeRobotDataset(repo_id=repo_id, download_videos=False)
    FPS = old_dataset.meta.fps
    ROBOT_TYPE = old_dataset.meta.robot_type
    old_features = deepcopy(old_dataset.features)
    total_frames = len(old_dataset)
    print(f"✓ 数据集加载完成: {total_frames} 帧")

    # 构建新 features，只保留 action + observation.state + images
    new_features = deepcopy(old_features)
    for k in list(new_features.keys()):
        if k in ["joint_action", "observation.joint_state", "timestamp", "task_index", "episode_index", "index", "frame_index"]:
            new_features.pop(k)

    # 确保action/state使用统一名字
    new_features["action"] = old_features["action"]
    new_features["observation.state"] = old_features["observation.state"]

    # 新数据集 repo_id
    target_repo_id = repo_id + "_ecaecs"

    # 创建新数据集
    new_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        features=new_features,
        fps=FPS,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=4
    )
    print(f"✓ 创建新数据集: {target_repo_id}")

    # --------------------------
    # 逐帧转换并写入
    # --------------------------
    for idx, sample in enumerate(old_dataset):
        new_sample = {}

        # 复制除action/state外的字段
        for k, v in sample.items():
            if k in ["action", "joint_action", "observation.state", "observation.joint_state",
                     "timestamp", "task_index", "episode_index", "index", "frame_index"]:
                continue
            if k.startswith("observation.images"):
                new_sample[k] = v.permute(1,2,0)
            else:
                new_sample[k] = v

        # --------------------------
        # action: base -> camera
        # --------------------------
        action = sample["action"].numpy() if isinstance(sample["action"], torch.Tensor) else np.array(sample["action"])
        pos_a = action[:3]
        rotvec_a = action[3:6]
        gripper_a = action[6]
        action_cam = transform_pose_base_to_cam(pos_a, rotvec_a, gripper_a, T_cam_base)
        new_sample["action"] = torch.from_numpy(action_cam).float()

        # --------------------------
        # state: base -> camera
        # --------------------------
        state = sample["observation.state"].numpy() if isinstance(sample["observation.state"], torch.Tensor) else np.array(sample["observation.state"])
        pos_s = state[:3]
        rotvec_s = state[3:6]
        gripper_s = state[6] if len(state) > 6 else 0.0
        state_cam = transform_pose_base_to_cam(pos_s, rotvec_s, gripper_s, T_cam_base)
        new_sample["observation.state"] = torch.from_numpy(state_cam).float()

        # --------------------------
        # 写入新数据集
        # --------------------------
        new_dataset.add_frame(new_sample)

        if (idx + 1) % 500 == 0:
            print(f"  已处理 {idx+1}/{total_frames} 帧")
            new_dataset.save_episode()
            new_dataset.episode_buffer = None

    # 保存剩余帧
    if new_dataset.episode_buffer and new_dataset.episode_buffer["size"] > 0:
        new_dataset.save_episode()
        new_dataset.episode_buffer = None
    new_dataset.finalize()

    new_dataset.finalize()
    print(f"✅ 数据集转换完成: {target_repo_id}")


# ============================================================
# 批量处理三个数据集
# ============================================================
if __name__ == "__main__":
    for repo_id, T_cam_base in zip(DATASETS, T_CAM_BASE_LIST):
        process_dataset(repo_id, T_cam_base)
