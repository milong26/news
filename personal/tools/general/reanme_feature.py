"""
处理单个repo_id
1. 选择合适的类型ee/j
2. 处理camera：分别计算偏移和旋转
3. 处理delta
4. 保存最后的数据集
"""

import torch
from copy import deepcopy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R

# --------------------------
# 用户配置
# --------------------------
SOURCE_REPO_ID = "test_one/ee"  # 原始数据集 repo_id
TARGET_REPO_ID = None            # None 自动生成

# 必须指定ee还是joint
# 后面可以选择 "camera""delta"
TARGET_ACTION = ["ee", "camera",]
TARGET_STATE  = ["joint", "camera"]

# --------------------------
# 坐标系转换函数
# --------------------------
HF_LEROBOT_HOME = Path("/home/qwe/.cache/huggingface/lerobot")

def load_camera_T(repo_id: str) -> np.ndarray:
    """从 repo meta 读取 T_base_to_camera"""
    path = HF_LEROBOT_HOME / repo_id / "meta" / "camera_T.json"
    with open(path) as f:
        data = json.load(f)
    return np.array(data["T_base_to_camera"])

def transform_position_rotation_to_camera(pos, rotvec, T_base_to_camera):
    """
    pos: (3,) 位置向量
    rotvec: (3,) 旋转向量 (axis-angle)
    T_base_to_camera: 4x4 numpy
    返回转换后的 pos, rotvec
    """
    # 位置
    pos_cam = T_base_to_camera[:3, :3] @ pos + T_base_to_camera[:3, 3]
    
    # 旋转
    R_ee = R.from_rotvec(rotvec).as_matrix()
    R_cam = T_base_to_camera[:3, :3] @ R_ee
    rotvec_cam = R.from_matrix(R_cam).as_rotvec()
    
    return pos_cam, rotvec_cam

def transform_to_camera(arr: np.ndarray):
    """
    处理 arr: 末端执行器状态 [x,y,z,wx,wy,wz,gripper]
    转换到相机坐标系
    """
    arr = np.array(arr, dtype=np.float32)
    if arr.shape[-1] == 7:  # ee状态
        pos, rotvec, grip = arr[:3], arr[3:6], arr[6]
        pos_cam, rotvec_cam = transform_position_rotation_to_camera(pos, rotvec, T_cam)
        arr_cam = np.concatenate([pos_cam, rotvec_cam, [grip]], axis=0)
        return arr_cam
    else:
        raise ValueError("不是ee数据，处理不了")
        # 对普通 3D 坐标处理
        shape = arr.shape
        arr_flat = arr.reshape(-1, 3)
        arr_h = np.hstack([arr_flat, np.ones((arr_flat.shape[0], 1))])
        arr_cam = (T_cam @ arr_h.T).T[:, :3]
        return arr_cam.reshape(shape)

# --------------------------
# 解析目标特征列表
# --------------------------
def parse_target_format(options: list[str]):
    options = [o.lower() for o in options]
    type_options = [o for o in options if o in ("ee","joint")]
    if len(type_options) != 1:
        raise ValueError(f"目标类型必须指定且只能有一个 'ee' 或 'joint', 当前: {options}")
    
    return {
        "type": type_options[0],
        "camera": "camera" in options,
        "delta": "delta" in options
    }

action_info = parse_target_format(TARGET_ACTION)
state_info  = parse_target_format(TARGET_STATE)

# --------------------------
# 功能函数
# --------------------------
def get_source_key(kind: str, typ: str):
    return {
        ("action","ee")  : "action",
        ("action","joint"): "joint_action",
        ("state","ee")   : "observation.state",
        ("state","joint"): "observation.joint_state"
    }[(kind, typ)]

def convert_frame(arr, T=None, delta=False, prev=None):
    """arr: torch.Tensor, 返回 torch.Tensor"""
    arr_np = arr.numpy() if isinstance(arr, torch.Tensor) else np.array(arr, dtype=np.float32)
    
    if T is not None:
        if arr_np.shape[-1] == 7:
            pos, rotvec, grip = arr_np[:3], arr_np[3:6], arr_np[6]
            pos_cam, rotvec_cam = transform_position_rotation_to_camera(pos, rotvec, T)
            arr_np = np.concatenate([pos_cam, rotvec_cam, [grip]], axis=0)
        else:
            # 对普通 3D 坐标处理
            shape = arr_np.shape
            arr_flat = arr_np.reshape(-1, 3)
            arr_h = np.hstack([arr_flat, np.ones((arr_flat.shape[0], 1))])
            arr_np = (T @ arr_h.T).T[:, :3].reshape(shape)
    
    if delta and prev is not None:
        arr_np = arr_np - prev
    
    return torch.tensor(arr_np, dtype=torch.float32)

# --------------------------
# 加载旧数据集
# --------------------------
old_dataset = LeRobotDataset(repo_id=SOURCE_REPO_ID)
FPS, ROBOT_TYPE = old_dataset.meta.fps, old_dataset.meta.robot_type
old_features = deepcopy(old_dataset.features)

source_action_key = get_source_key("action", action_info['type'])
source_state_key  = get_source_key("state", state_info['type'])

target_action_key = "action"
target_state_key  = "observation.state"

if TARGET_REPO_ID is None:
    TARGET_REPO_ID = f"{SOURCE_REPO_ID}_{'_'.join(TARGET_ACTION)}_{'_'.join(TARGET_STATE)}"

# --------------------------
# 坐标系转换矩阵（如果需要）
# --------------------------
T_cam = load_camera_T(SOURCE_REPO_ID) if (action_info['camera'] or state_info['camera']) else None

# --------------------------
# 构建新 features
# --------------------------
new_features = deepcopy(old_features)
for k in ["action", "joint_action", "observation.state", "observation.joint_state",
          'timestamp', 'task_index', 'episode_index', 'index', 'frame_index']:
    if k in new_features:
        new_features.pop(k)
new_features[target_action_key] = old_features[source_action_key]
new_features[target_state_key]  = old_features[source_state_key]

# --------------------------
# 创建新数据集
# --------------------------
new_dataset = LeRobotDataset.create(
    repo_id=TARGET_REPO_ID,
    features=new_features,
    fps=FPS,
    robot_type=ROBOT_TYPE,
    use_videos=True,
    image_writer_threads=4
)

# --------------------------
# 拷贝数据并处理 camera / delta
# --------------------------
prev_action = None
prev_state  = None

for idx, sample in enumerate(old_dataset):
    new_sample = {}

    # 拷贝其他字段（非 action/state）
    for k, v in sample.items():
        if k not in ["action", "joint_action", "observation.state", "observation.joint_state",
                     'timestamp', 'task_index', 'episode_index', 'index', 'frame_index']:
            new_sample[k] = v.permute(1,2,0) if k.startswith("observation.images") else v

    # 转换 action/state
    new_sample[target_action_key] = convert_frame(
        sample[source_action_key],
        T=T_cam if action_info['camera'] else None,
        delta=action_info['delta'],
        prev=prev_action
    )
    new_sample[target_state_key] = convert_frame(
        sample[source_state_key],
        T=T_cam if state_info['camera'] else None,
        delta=state_info['delta'],
        prev=prev_state
    )

    prev_action = new_sample[target_action_key].clone()
    prev_state  = new_sample[target_state_key].clone()

    # 添加到新 dataset
    new_dataset.add_frame(new_sample,use_origin_dataset=True)

    # 每 1000 帧保存一次
    if (idx + 1) % 1000 == 0:
        new_dataset.save_episode()
        new_dataset.episode_buffer = None

# 保存剩余帧
if new_dataset.episode_buffer and new_dataset.episode_buffer["size"] > 0:
    new_dataset.save_episode()
    new_dataset.episode_buffer = None

new_dataset.finalize()
print(f"✅ 数据集转换完成: {TARGET_REPO_ID}")
