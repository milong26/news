"""
pip install pupil_apriltags
pip install scipy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pupil_apriltags import Detector
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R
import traceback
import sys
import gc


def get_intrinsics():
    """获取相机内参"""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    camera_intrinsics = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy
    }
    
    pipeline.stop()
    
    print(f"✓ 相机内参: fx={camera_intrinsics['fx']:.2f}, fy={camera_intrinsics['fy']:.2f}, "
          f"cx={camera_intrinsics['cx']:.2f}, cy={camera_intrinsics['cy']:.2f}")
    
    return camera_intrinsics


def state_to_matrix(state):
    """
    将状态向量转换为4x4变换矩阵
    state: [x, y, z, qx, qy, qz, qw]
    """
    pos = state[:3]
    quat = state[3:7]
    
    # 检查四元数是否有效
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 0.01:
        raise ValueError(f"四元数范数过小: {quat_norm}")
    
    # 归一化四元数
    quat = quat / quat_norm
    
    rot = R.from_quat(quat).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def detect_apriltag_pose(
    image,
    detector,
    tag_size=0.02,
    tag_id=0,
    camera_intrinsics=None,
    max_z_cos=0.97,
    min_decision_margin=50
):
    """
    检测AprilTag并返回位姿
    
    返回:
        T_cam_tag (4x4 numpy array) 或 None
    """
    # 转换为灰度图
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"))
    else:
        gray = np.array(Image.fromarray(image).convert("L"))

    # 检查图像尺寸
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return None

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # 检测AprilTag
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=tag_size
    )

    # 查找目标tag
    for det in detections:
        if det.tag_id != tag_id:
            continue

        # 质量检查
        if det.hamming > 0:
            return None
            
        if det.decision_margin < min_decision_margin:
            return None

        # 法向量检查(避免退化视角)
        z_cam = det.pose_R[:, 2]
        if abs(z_cam[2]) > max_z_cos:
            return None

        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = det.pose_R
        T[:3, 3] = det.pose_t.squeeze()
        
        return T

    return None


def process_batch(
    dataset,
    start_idx,
    end_idx,
    image_key,
    state_key,
    camera_intrinsics,
    tag_size,
    tag_id,
    T_tag_ee
):
    """
    处理一批数据
    
    返回:
        batch_Ts: 该批次的有效变换矩阵列表
        valid_count: 有效帧数
    """
    # 创建检测器(每批都创建新的)
    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    batch_Ts = []
    valid_count = 0
    
    try:
        for idx in range(start_idx, end_idx):
            if idx >= len(dataset):
                break
                
            try:
                # 获取数据
                item = dataset[idx]
                
                # 获取图像
                img = item[image_key]
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).numpy()
                    if img.max() <= 1:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    img = Image.fromarray(img)
                elif not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img).astype(np.uint8))
                
                # 获取状态
                state = item[state_key]
                if isinstance(state, torch.Tensor):
                    state = state.numpy()
                state = np.array(state, dtype=np.float64)
                
                if len(state) < 7:
                    continue
                
                # 转换为变换矩阵
                T_ee_base = state_to_matrix(state)
                
                # 检测AprilTag
                T_cam_tag = detect_apriltag_pose(
                    img,
                    detector,
                    tag_size=tag_size,
                    tag_id=tag_id,
                    camera_intrinsics=camera_intrinsics
                )
                
                if T_cam_tag is None:
                    continue
                
                # 计算T_cam_base
                T_cam_base = T_cam_tag @ np.linalg.inv(T_tag_ee) @ np.linalg.inv(T_ee_base)
                
                # 检查结果是否有效
                if np.any(np.isnan(T_cam_base)) or np.any(np.isinf(T_cam_base)):
                    continue
                
                batch_Ts.append(T_cam_base)
                valid_count += 1
                
                # 简洁输出
                if (idx - start_idx + 1) % 50 == 0:
                    print(f"  进度: {idx - start_idx + 1}/{end_idx - start_idx}, 有效: {valid_count}")
                    sys.stdout.flush()
                
            except Exception as e:
                # 单帧错误不影响其他帧
                continue
                
    finally:
        # 确保检测器被删除
        del detector
        gc.collect()  # 强制垃圾回收
    
    return batch_Ts, valid_count


def process_dataset(
    repo_id,
    image_key="observation.images.side",
    state_key="observation.state",
    tag_size=0.02,
    tag_id=0,
    T_tag_ee=None,
    batch_size=500
):
    """
    批量处理数据集并计算手眼标定矩阵
    
    参数:
        repo_id: 数据集路径
        image_key: 图像数据的键
        state_key: 状态数据的键
        tag_size: AprilTag尺寸(米)
        tag_id: AprilTag ID
        T_tag_ee: tag到末端执行器的固定变换(4x4矩阵)
        batch_size: 每批处理的帧数
    
    返回:
        Ts: 所有有效的T_cam_ee矩阵
    """
    # 加载数据集
    print(f"\n正在加载数据集: {repo_id}")
    try:
        dataset = LeRobotDataset(repo_id=repo_id, download_videos=False)
        print(f"✓ 数据集加载完成,共 {len(dataset)} 帧")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        traceback.print_exc()
        return None

    # 获取相机内参
    try:
        camera_intrinsics = get_intrinsics()
    except Exception as e:
        print(f"❌ 获取相机内参失败: {e}")
        traceback.print_exc()
        return None

    # 默认T_tag_ee为单位矩阵
    if T_tag_ee is None:
        T_tag_ee = np.eye(4)
        print("⚠ 使用默认T_tag_ee(单位矩阵),请根据实际安装修改")

    # 计算批次数
    total_frames = len(dataset)
    num_batches = (total_frames + batch_size - 1) // batch_size
    
    print(f"\n开始批量处理:")
    print(f"  总帧数: {total_frames}")
    print(f"  批次大小: {batch_size}")
    print(f"  批次数量: {num_batches}")
    print("=" * 60)

    all_Ts = []
    total_valid = 0
    
    # 逐批处理
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_frames)
        
        print(f"\n📦 批次 {batch_idx + 1}/{num_batches}")
        print(f"   处理帧: {start_idx} - {end_idx - 1}")
        
        try:
            batch_Ts, valid_count = process_batch(
                dataset=dataset,
                start_idx=start_idx,
                end_idx=end_idx,
                image_key=image_key,
                state_key=state_key,
                camera_intrinsics=camera_intrinsics,
                tag_size=tag_size,
                tag_id=tag_id,
                T_tag_ee=T_tag_ee
            )
            
            all_Ts.extend(batch_Ts)
            total_valid += valid_count
            
            print(f"   ✓ 本批有效: {valid_count}/{end_idx - start_idx}")
            print(f"   累计有效: {total_valid}/{end_idx}")
            
            # 显式释放内存
            del batch_Ts
            gc.collect()
            
        except KeyboardInterrupt:
            print("\n\n⚠ 用户中断")
            break
        except Exception as e:
            print(f"   ✗ 批次处理失败: {e}")
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print(f"✓ 全部处理完成: {total_valid}/{total_frames} 帧有效 ({100*total_valid/total_frames:.1f}%)")
    print("=" * 60 + "\n")
    
    if total_valid == 0:
        return None
    
    return np.stack(all_Ts)


def visualize_results(Ts):
    """可视化标定结果"""
    if Ts is None or len(Ts) == 0:
        print("❌ 没有有效数据,无法绘图")
        return

    print(f"正在绘制 {len(Ts)} 个有效标定矩阵...")
    
    try:
        # 计算平均值和标准差
        T_mean = Ts.mean(axis=0)
        T_std = Ts.std(axis=0)

        # 绘制16个子图(4x4矩阵的每个元素)
        fig, axs = plt.subplots(4, 4, figsize=(16, 12))
        fig.suptitle(f'Hand-Eye Calibration Results (n={len(Ts)})', fontsize=16)
        
        for i in range(4):
            for j in range(4):
                ax = axs[i, j]
                values = Ts[:, i, j]
                
                # 绘制数据点和平均线
                ax.plot(values, marker='o', markersize=2, linewidth=0.5, alpha=0.6)
                ax.axhline(T_mean[i, j], color='r', linestyle='--', linewidth=2, label='Mean')
                ax.axhline(T_mean[i, j] + T_std[i, j], color='orange', linestyle=':', alpha=0.5)
                ax.axhline(T_mean[i, j] - T_std[i, j], color='orange', linestyle=':', alpha=0.5)
                
                ax.set_title(f'T[{i},{j}]', fontsize=10)
                ax.set_xlabel('Frame', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6)
                
                # 显示统计信息
                ax.text(0.02, 0.98, f'μ={T_mean[i,j]:.4f}\nσ={T_std[i,j]:.4f}',
                       transform=ax.transAxes, fontsize=7,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.show()

        # 打印结果
        print("\n" + "="*60)
        print("平均 T_cam_base 矩阵:")
        print("="*60)
        print(T_mean)
        print("\n标准差:")
        print(T_std)
        print("\n最大标准差元素: T[{},{}] = {:.6f}".format(
            *np.unravel_index(T_std.argmax(), T_std.shape),
            T_std.max()
        ))
        print("="*60)
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        traceback.print_exc()


def main():
    """主函数"""
    try:
        # 配置参数
        repo_id = "collect_data/third_camera"  # 数据集路径
        image_key = "observation.images.side"  # 图像键
        state_key = "observation.state"  # 状态键
        tag_size = 0.02  # AprilTag尺寸(米)
        tag_id = 0  # AprilTag ID
        batch_size = 500  # 批次大小(根据内存调整: 100-1000)
        
        # Tag到末端执行器的固定变换(根据实际安装修改)
        T_tag_ee = np.eye(4)
        # 例如: T_tag_ee[:3, 3] = [0.01, 0.02, 0.03]  # x, y, z偏移
        
        # 处理数据集
        Ts = process_dataset(
            repo_id=repo_id,
            image_key=image_key,
            state_key=state_key,
            tag_size=tag_size,
            tag_id=tag_id,
            T_tag_ee=T_tag_ee,
            batch_size=batch_size
        )
        
        # 可视化结果
        visualize_results(Ts)
        
    except Exception as e:
        print(f"\n❌ 程序异常退出: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()