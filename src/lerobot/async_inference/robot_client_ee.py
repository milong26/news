
from pynput import keyboard
import numpy as np
import cv2
import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)

# 运动学相关
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
# feature相关
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features,create_initial_features
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.utils import make_robot_action

# 重新搞，不能用pipeline
from lerobot.robots.so100_follower.robot_kinematic_processor import compute_forward_kinematics_joints_to_ee

import json

from scipy.spatial.transform import Rotation
import numpy as np

from scipy.spatial.transform import Rotation as R
def compute_robot_joints_from_ee(
    action: dict,
    observation: dict,
    kinematics_solver,
    motor_names,
    q_curr=None,
    initial_guess_current_joints=True,
    ee_bounds=None,
    max_ee_step_m=0.10
):
    """
    将 end-effector 指令转为关节位置（仿照 pipeline 的功能）。
    
    Args:
        action: dict, 包含 'ee.x', 'ee.y', 'ee.z', 'ee.wx', 'ee.wy', 'ee.wz', 'ee.gripper_pos'
        observation: dict, 当前关节状态
        kinematics_solver: RobotKinematics 对象
        motor_names: list, 机器人关节名称
        q_curr: np.ndarray, 上一次关节位置（可选）
        initial_guess_current_joints: bool, 是否使用当前关节作为 IK 初值
        ee_bounds: dict, {"min": [...], "max": [...]}, end-effector 坐标限制
        max_ee_step_m: float, 最大允许位移
    Returns:
        dict: 处理后的关节指令
        np.ndarray: 更新后的 q_curr
    """

    # --- 1. EEBoundsAndSafety ---
    x, y, z = action['ee.x'], action['ee.y'], action['ee.z']
    if ee_bounds:
        for i, coord in enumerate([x, y, z]):
            if coord < ee_bounds["min"][i] or coord > ee_bounds["max"][i]:
                raise ValueError(f"EE target {coord} exceeds bounds {ee_bounds}")
    
    # 可选: 检查最大步长 (这里简化, 可按需要实现)
    
    # --- 2. 准备当前关节状态 ---
    q_raw = np.array([float(v) for k, v in observation.items() if k.endswith(".pos")], dtype=float)
    if initial_guess_current_joints or q_curr is None:
        q_curr = q_raw

    # --- 3. 构建目标位姿 ---
    wx, wy, wz = action['ee.wx'], action['ee.wy'], action['ee.wz']
    gripper_pos = action['ee.gripper_pos']

    t_des = np.eye(4)
    t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
    t_des[:3, 3] = [x, y, z]

    # --- 4. 计算 IK ---
    q_target = kinematics_solver.inverse_kinematics(q_curr, t_des)
    q_curr = q_target  # 更新状态

    # --- 5. 填充关节指令 ---
    joint_action = {}
    for i, name in enumerate(motor_names):
        if name != "gripper":
            joint_action[f"{name}.pos"] = float(q_target[i])
        else:
            joint_action["gripper.pos"] = float(gripper_pos)

    return joint_action, q_curr


# 处理state
def process_state_base_to_cam(
    obs_processed_base: dict,
    T_cam_base: np.ndarray,
):
    """
    输入:
        obs_processed_base:
            EE pose in BASE frame
            {
              "x","y","z","rx","ry","rz","gripper"
            }
        T_cam_base: 4x4

    输出:
        torch.Tensor (7,)
        EE pose in CAMERA frame
    """
    pos_base = np.array([
        obs_processed_base["ee.x"],
        obs_processed_base["ee.y"],
        obs_processed_base["ee.z"],
    ])

    rotvec_base = np.array([
        obs_processed_base["ee.wx"],
        obs_processed_base["ee.wy"],
        obs_processed_base["ee.wz"],
    ])

    gripper = obs_processed_base.get("ee.gripper_pos", 0.0)

    # --- base → cam ---
    R_cam_base = T_cam_base[:3, :3]
    t_cam_base = T_cam_base[:3, 3]

    pos_cam = R_cam_base @ pos_base + t_cam_base

    R_base = R.from_rotvec(rotvec_base).as_matrix()
    R_cam = R_cam_base @ R_base
    rotvec_cam = R.from_matrix(R_cam).as_rotvec()

    state_cam = np.concatenate([pos_cam, rotvec_cam, [gripper]])

    return torch.from_numpy(state_cam).float()

# 处理action
def process_action_cam_to_base(
    act_processed_policy: dict,
    T_cam_base: np.ndarray,
):
    """
    输入:
        act_processed_policy:
            EE action in CAMERA frame
            {
              "x","y","z","rx","ry","rz","gripper"
            }

    输出:
        dict
        EE action in BASE frame
    """
    pos_cam = np.array([
        act_processed_policy["ee.x"],
        act_processed_policy["ee.y"],
        act_processed_policy["ee.z"],
    ])

    rotvec_cam = np.array([
        act_processed_policy["ee.wx"],
        act_processed_policy["ee.wy"],
        act_processed_policy["ee.wz"],
    ])

    gripper = act_processed_policy.get("ee.gripper_pos", 0.0)

    # --- cam → base ---
    R_cam_base = T_cam_base[:3, :3]
    t_cam_base = T_cam_base[:3, 3]

    R_base_cam = R_cam_base.T
    t_base_cam = -R_base_cam @ t_cam_base

    pos_base = R_base_cam @ pos_cam + t_base_cam

    R_cam = R.from_rotvec(rotvec_cam).as_matrix()
    R_base = R_base_cam @ R_cam
    rotvec_base = R.from_matrix(R_base).as_rotvec()

    return {
        "ee.x": pos_base[0],
        "ee.y": pos_base[1],
        "ee.z": pos_base[2],
        "ee.wx": rotvec_base[0],
        "ee.wy": rotvec_base[1],
        "ee.wz": rotvec_base[2],
        "ee.gripper_pos": gripper,
    }


firsttime=True
class RobotClientEE:
    prefix = "robot_client_ee"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        # 运动学相关
        # 1. 求解器
        self.kinematics_solver = RobotKinematics(
            urdf_path="SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
            target_frame_name="gripper_frame_link",
            joint_names=list(self.robot.bus.motors.keys()),
        )
        # 2.逆向的计算
        self.robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            [
                EEBoundsAndSafety(
                    end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                    max_ee_step_m=0.10,
                ),
                InverseKinematicsEEToJoints(
                    kinematics=self.kinematics_solver,
                    motor_names=list(self.robot.bus.motors.keys()),
                    initial_guess_current_joints=True,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        # 3. 正向运算
        self.robot_joints_to_ee_pose_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
            steps=[
                ForwardKinematicsJointsToEE(kinematics=self.kinematics_solver, motor_names=list(self.robot.bus.motors.keys()))
            ],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )
        # 新增ee_feature
        self.all_feature=aggregate_pipeline_dataset_features(
            pipeline=self.robot_joints_to_ee_pose_processor,
            initial_features=create_initial_features(observation=self.robot.observation_features),
            use_videos=True,
            # TODO true?
        ),
        # User for now should be explicit on the feature keys that were used for record
        # Alternatively, the user can pass the processor step that has the right features
        self.action_feature=aggregate_pipeline_dataset_features(
            pipeline=make_default_teleop_action_processor(),
            initial_features=create_initial_features(
                action={
                    f"ee.{k}": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                    for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
                }
            ),
            use_videos=True,
        ),

        # 原来的feature：生成 LeRobot dataset 的特征字典，规则非常固定
        # lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        # 原来feature
        # 'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']}, 
        # 'observation.images.wrist': {'dtype': 'image', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']},
        # 'observation.images.side': {'dtype': 'image', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}}
        # 重新整理policy feature，参考evaluate的feature
        # 没有action的要求？
        # print("旧的",lerobot_features)
        # self.state_feature[0].pop('action', None)
        # print("新的",self.state_feature[0])
        state_only = {k: v for k, v in self.all_feature[0].items() if k != 'action'}
        lerobot_features=state_only
        # 为什么dtype变成了video....
        # {'observation.state': {'dtype': 'float32', 'shape': (7,), 'names': ['ee.x', 'ee.y', 'ee.z', 'ee.wx', 'ee.wy', 'ee.wz', 'ee.gripper_pos']},
        # 'observation.images.wrist': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}, 
        # 'observation.images.side': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}}

        
        self.q_curr=np.array(
            [
                self.robot.get_only_state_obs()[f"{name}.pos"]
                for name in self.robot.bus.motors.keys()
            ],
            dtype=float,
        )
        
        
        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address
        # 检验这个policy_config是否正确
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            # rename_map=config.rename_map
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing
       # 监听键盘输入->的时候清空action缓存
        threading.Thread(target=self._listen_clear_key,daemon=True).start()
        # 等待时间
        self.pause_until=0
        # 为了记录state和action增加的
        self.logged_timestamps = []
        self.logged_actions = []  # list of dicts
        self.logged_joint_states = []  # list of dicts


        # actions_path = "robot_actions.json"
        # with open(actions_path, "r") as f:
        #     self.preloaded_actions: list[dict[str, float]] = json.load(f)

        # if len(self.preloaded_actions) == 0:
        #     raise ValueError("robot_actions.json is empty!")
        self.preloaded_action_idx: int = 0
        # self.logger.info(
        #     f"Loaded {len(self.preloaded_actions)} pre-recorded robot actions"
        # )

        self.T_cam_base_new=  np.array([
            [-0.36754856, -0.05893811, -0.92813488,  0.14663559],
            [ 0.22117079,  0.96381803, -0.18070615,  -0.03928965],
            [ 0.90332250, -0.29996365, -0.40121454,  0.23130315],
            [ 0.0,         0.0,         0.0,         1.0       ]
        ], dtype=np.float64)




    def _listen_clear_key(self):
        def on_press(key):
            try:
                if key==keyboard.Key.right:
                    self.clear_action_queue()
                if key==keyboard.Key.left:
                    self.running=False
            except AttributeError:
                self.logger.info("键盘出问题")
        # with keyboard.Listener(on_press=on_press) as listener:
        #     listener.join()
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        self.shutdown_event.wait()
        listener.stop()

    def clear_action_queue(self,pause_seconds:float=3.0):
        with self.action_queue_lock:
            # while not self.action_queue.empty():
                # self.action_queue.get_nowait()
                # 直接新建队列
                self.action_queue = Queue()
        self.must_go.set()
        self.logger.info("清空了actionqueue，顺便设置了mustgo，停个几秒")
        # 要设置control_loop里面暂停
        self.pause_until=time.time()+pause_seconds

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()
        try:
            self.start_barrier.abort()
        except:
            pass


        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            # 右键清空，暂停几秒
            if time.time()<self.pause_until:
                time.sleep(0.1)
                continue
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action
    
    def _action_tensor_to_action_dict_ee(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action



    def transfer_cam_to_base(policy_action, T_base_to_cam):
        """
        将 policy 推理得到的【相机坐标系下的绝对末端位姿】
        转换为【机械臂 base 坐标系下的绝对末端位姿】

        参数
        ----
        policy_action : dict
            policy 输出的 action，必须包含：
            policy_action["pose"] = [x, y, z, qx, qy, qz, qw]  (camera frame)

        T_base_to_cam : np.ndarray (4,4)
            相机外参：base -> camera

        返回
        ----
        base_policy_action : dict
            与 policy_action 结构一致，但 pose 已转换到 base 坐标系
        """

        # -------- sanity check --------
        assert "pose" in policy_action, "policy_action must contain 'pose'"
        pose_cam = np.asarray(policy_action["pose"])
        assert pose_cam.shape[0] == 7, "pose must be [x,y,z,qx,qy,qz,qw]"

        # -------- cam pose -> T --------
        T_cam_to_ee = np.eye(4)
        T_cam_to_ee[:3, 3] = pose_cam[:3]
        T_cam_to_ee[:3, :3] = R.from_quat(pose_cam[3:7]).as_matrix()

        # -------- base <- cam --------
        T_cam_to_base = np.linalg.inv(T_base_to_cam)
        T_base_to_ee = T_cam_to_base @ T_cam_to_ee

        # -------- T -> pose --------
        pose_base = np.zeros(7)
        pose_base[:3] = T_base_to_ee[:3, 3]
        pose_base[3:7] = R.from_matrix(T_base_to_ee[:3, :3]).as_quat()

        # -------- copy action --------
        base_policy_action = dict(policy_action)   # shallow copy is enough
        base_policy_action["pose"] = pose_base

        return base_policy_action


    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        if not self.running:
            return None

        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start
        # 难道是这里的get_observation执行次数太多了？
        
        policy_action=timed_action.get_action()
        # 调用相机处理
        # base_policy_action=self.transfer_cam_to_base(policy_action)
        # policy_action(tensor->dict)
        act_processed_policy: RobotAction = make_robot_action(policy_action, self.action_feature[0])

        act_processed_policy = process_action_cam_to_base(
            act_processed_policy,
            self.T_cam_base_new
        )

        # act_processed_policy=self._action_tensor_to_action_dict(policy_action)
        # eeaction->joint_action
        joint_state=self.robot.get_only_state_obs()
        # print("得到的joint_state是",joint_state)
        """
        自己写,不用pipeline
        """
        # 为什么算出来又不一样了???
        # robot_action_to_send = self.robot_ee_to_joints_processor((act_processed_policy, joint_state))
        robot_action_to_send, self.q_curr = compute_robot_joints_from_ee(
            action=act_processed_policy,
            observation=joint_state,
            kinematics_solver=self.kinematics_solver,
            motor_names=list(self.robot.bus.motors.keys()),
            q_curr=self.q_curr,
            # q_curr=None,
            initial_guess_current_joints=False,
            # initial_guess_current_joints=True,
            ee_bounds={"min": [-1, -1, -1], "max": [1, 1, 1]},
            max_ee_step_m=0.10,
        )
        
        
        _performed_action = self.robot.send_action(
            robot_action_to_send
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )
        # timestamp = time.time()
        # self.logged_timestamps.append(timestamp)
        self.logged_actions.append(_performed_action)  # joint_action 是 dict
        self.logged_joint_states.append(joint_state)  # joint_state 是 dict
        # print("return之前",self.robot.get_only_state_obs()["shoulder_pan.pos"])

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    # 获取observation并发送
    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task
            # 经过pipeline变成ee形式的state，从而发送给服务器做eestate->eeaction推理
            # 这里？
            # 如果只处理state怎么样？不处理image
            keys_to_keep = [
                'shoulder_pan.pos', 
                'shoulder_lift.pos', 
                'elbow_flex.pos', 
                'wrist_flex.pos', 
                'wrist_roll.pos', 
                'gripper.pos'
            ]
            raw_observation_state= {k: raw_observation[k] for k in keys_to_keep}
            motor_names = list(self.robot.bus.motors.keys()) 
            obs_processed = compute_forward_kinematics_joints_to_ee(
                joints=raw_observation_state,
                kinematics=self.kinematics_solver,
                motor_names=motor_names
            )
            # obs_processed = self.robot_joints_to_ee_pose_processor(raw_observation_state)
            image_keys = ['wrist', 'side','task']
            for k in image_keys:
                obs_processed[k] = raw_observation[k]
            # with open("obs_saved.pkl", "wb") as f:
            #     pickle.dump(obs_processed, f)
            # with open('obs_saved.pkl', 'rb') as f:
            #     obs_processed = pickle.load(f)
            # # 1. 先加载之前计算好的单应性矩阵 H
            # H = np.load("homography.npy")  # shape (3,3)

            # # 2. 假设你在循环里获取obs后，直接对camera1的图像做变换
            # if "side" in raw_observation:
            #     img = raw_observation["side"]  # 这里 img 是 numpy array 格式
            #     h, w = img.shape[:2]
            #     # 将新位置图像映射到旧位置视角
            #     warped_img = cv2.warpPerspective(img, H, (w, h))
            #     raw_observation["side"] = warped_img  # 覆盖原来的图像

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=obs_processed,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return obs_processed

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
            # print("执行完control_loop_action",self.robot.get_only_state_obs()["shoulder_pan.pos"])

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # print("执行完logger",self.robot.get_only_state_obs()["shoulder_pan.pos"])
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))
            print("等了", 0.03 - (time.perf_counter() - control_loop_start),self.robot.get_only_state_obs())
            # time.sleep(0.03)
            # print("执行完sleep0.03之后的state",self.robot.get_only_state_obs()["shoulder_pan.pos"])

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClientEE(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)



        finally:
            action_receiver_thread.join(timeout=2.0)
            client.stop()
            # action_receiver_thread.join()
            # print(client.logged_actions)
            # plot_actions(client)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")
        


import matplotlib.pyplot as plt
def plot_actions(client: RobotClientEE):
    times = client.logged_timestamps
    actions = client.logged_actions
    joint_states = client.logged_joint_states

    keys = list(actions[0].keys())  # action dict 的 key
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=True)

    for i, key in enumerate(keys):
        ax = axes[i]
        y_action = [a[key] for a in actions]
        y_joint = [s[key] for s in joint_states]  # joint_state 对应 key
        ax.plot(times, y_action, label='performed_action')
        ax.plot(times, y_joint, label='joint_state', linestyle='--')
        ax.set_ylabel(key)
        ax.legend()

    plt.xlabel('Time (cishu)')
    plt.show(block=True) 

if __name__ == "__main__":
    async_client()  # run the client
