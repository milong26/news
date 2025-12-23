import torch
from copy import deepcopy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --------------------------
# 用户配置
# --------------------------
SOURCE_REPO_ID = "test_one/ee"  # 原始数据集 repo_id
TARGET_XAXS = "eajs"            # eaes/jajs/eajs/jaes
TARGET_REPO_ID = None            # None 自动生成

# --------------------------
# 工具函数
# --------------------------
def parse_xaxs(xaxs: str):
    """
    解析 xaxs 字符串，返回 action_type/state_type
    'e' = ee形式, 'j' = joint形式
    """
    assert len(xaxs) == 4
    action_type = "ee" if xaxs[0] == "e" else "joint"
    state_type  = "ee" if xaxs[2] == "e" else "joint"
    return action_type, state_type

def get_source_key(kind: str, typ: str):
    """
    根据类型返回原始 feature 名
    """
    return {
        ("action","ee")  : "action",
        ("action","joint"): "joint_action",
        ("state","ee")   : "observation.state",
        ("state","joint"): "observation.joint_state"
    }[(kind, typ)]

# --------------------------
# 加载旧数据集
# --------------------------
old_dataset = LeRobotDataset(repo_id=SOURCE_REPO_ID)
FPS, ROBOT_TYPE = old_dataset.meta.fps, old_dataset.meta.robot_type
old_features = deepcopy(old_dataset.features)

# --------------------------
# 解析目标 xaxs
# --------------------------
action_type, state_type = parse_xaxs(TARGET_XAXS)
source_action_key = get_source_key("action", action_type)
source_state_key  = get_source_key("state", state_type)

# 新数据集中 action / state 固定名字
target_action_key = "action"
target_state_key  = "observation.state"

# 自动生成 repo_id
if TARGET_REPO_ID is None:
    TARGET_REPO_ID = f"{SOURCE_REPO_ID}_{TARGET_XAXS}"

# --------------------------
# 构建新 features
# --------------------------
# 先复制旧的所有 feature
new_features = deepcopy(old_features)

# 删除旧的 action/state feature
for k in ["action", "joint_action", "observation.state", "observation.joint_state",'timestamp', 'task_index', 'episode_index', 'index', 'frame_index']:
    if k in new_features:
        new_features.pop(k)

# 添加统一的 target action/state feature
new_features[target_action_key] = old_features[source_action_key]
new_features[target_state_key]  = old_features[source_state_key]

print("保留的 feature:", list(new_features.keys()))

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
# 拷贝数据
# --------------------------
for idx, sample in enumerate(old_dataset):
    new_sample = {}

    # 拷贝非 action/state 的字段
    for k, v in sample.items():
        if k not in ["action", "joint_action", "observation.state", "observation.joint_state",'timestamp', 'task_index', 'episode_index', 'index', 'frame_index']:
            new_sample[k] = v.permute(1,2,0) if k.startswith("observation.images") else v

    # 拷贝目标 action/state 到统一名字
    new_sample[target_action_key] = (
        sample[source_action_key].clone() if isinstance(sample[source_action_key], torch.Tensor) else torch.tensor(sample[source_action_key])
    )
    new_sample[target_state_key] = (
        sample[source_state_key].clone() if isinstance(sample[source_state_key], torch.Tensor) else torch.tensor(sample[source_state_key])
    )

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
