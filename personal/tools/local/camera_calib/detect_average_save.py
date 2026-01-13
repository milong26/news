import json
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import cv2
from pupil_apriltags import Detector

# ================== 用户参数 ==================
TAG_SIZE = 60.0  # mm
# base坐标系下的位置
# 正面
# TAG_3D_POINTS = {
#     0: [128, 163, 0],
#     1: [218, 163, 0],
#     2: [308, 163, 0],
#     3: [128, 73, 0],
#     4: [218, 73, 0],
#     5: [308, 73, 0],
# }
# 侧面
TAG_3D_POINTS = {
    0: [386, 72, 0],
    1: [386, -18, 0],
    2: [386, -108, 0],
    3: [296, 72, 0],
    4: [296, -18, 0],
    5: [296, -108, 0],
}
AXIS_LENGTH = 180         # base 坐标轴长度
TAG_AXIS_LENGTH = 30     # 每个 tag 坐标轴长度
# ==============================================

def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len=AXIS_LENGTH):
    """画 base 坐标轴或 tag 坐标轴"""
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ])
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    o, x, y, z = imgpts.reshape(-1, 2).astype(int)

    cv2.line(img, tuple(o), tuple(x), (0, 0, 255), 2)   # X 红
    cv2.line(img, tuple(o), tuple(y), (0, 255, 0), 2)   # Y 绿
    cv2.line(img, tuple(o), tuple(z), (255, 0, 0), 2)   # Z 蓝
    cv2.circle(img, tuple(o), 3, (0, 0, 0), -1)

def save_camera_T(repo_id: str, T_base_to_camera: np.ndarray):
    HF_LEROBOT_HOME = Path("/home/qwe/.cache/huggingface/lerobot")
    meta_dir = HF_LEROBOT_HOME / repo_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "camera_T.json"

    T_list = T_base_to_camera.tolist()
    with open(out_path, "w") as f:
        json.dump({"T_base_to_camera": T_list}, f, indent=4)

    print(f"[INFO] Saved T_base_to_camera to {out_path}")
    return out_path

def average_transform(rvec_list, tvec_list):
    """
    rvec_list: list of (3,1) or (3,) rotation vectors
    tvec_list: list of (3,1) or (3,) translation vectors
    返回：R_avg (3x3), t_avg (3,)
    """
    if len(rvec_list) == 0:
        raise ValueError("No transforms to average")

    # convert rvecs to rotation matrices and sum
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for rv in rvec_list:
        rv = np.array(rv).reshape(3, 1)
        R, _ = cv2.Rodrigues(rv)
        R_sum += R

    # SVD orthogonalization
    U, S, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        # fix reflection
        U[:, -1] *= -1
        R_avg = U @ Vt

    # average translation
    t_arr = np.array([np.array(t).reshape(3) for t in tvec_list], dtype=np.float64)
    t_avg = t_arr.mean(axis=0)

    return R_avg, t_avg

def main():
    REPO_ID = input("请输入 dataset repo_id: ").strip()
    print("按键说明：")
    print("  p : 开始/停止 采样；停止时会对已采样的 r/t 做平均并保存到 repo")
    print("  ESC : 退出程序（若有未保存的采样，会在退出前自动平均并保存）")
    print("请把至少 4 个已知 ID 的 AprilTag 放在视野中以便求解 base -> camera 变换。")

    # ---------- 获得Realsense ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(intr.coeffs)

    # ---------- Apriltag Detector ----------
    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1
    )

    print("开始标定预览（窗口中会显示提示），至少看到 4 个 tag 才能解算 base->camera。")

    # 采样相关
    recording = False
    saved_once = False
    collected_rvecs = []
    collected_tvecs = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detections = detector.detect(gray)

            obj_pts = []
            img_pts = []

            for det in detections:
                tag_id = det.tag_id

                # 画 tag 中心点
                c = tuple(det.center.astype(int))
                cv2.circle(img, c, 5, (0, 255, 255), -1)
                cv2.putText(img, f"id:{tag_id}", c,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if tag_id in TAG_3D_POINTS:
                    obj_pts.append(TAG_3D_POINTS[tag_id])
                    img_pts.append(det.center)

                # 每个 tag 的坐标轴（用 tag 四角 solvePnP）
                corners_2d = det.corners.astype(np.float32)
                corners_2d_int = det.corners.astype(int)
                for i, corner in enumerate(corners_2d_int):
                    x, y = corner
                    cv2.circle(img, (x, y), 5, (255, 0, 255), -1)
                    cv2.putText(img, f"{i+1}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                half_size = TAG_SIZE / 2
                # obj_tag = np.array([
                #     [-half_size, -half_size, 0],
                #     [ half_size,  -half_size, 0],
                #     [ half_size, half_size, 0],
                #     [-half_size, half_size, 0],
                # ], dtype=np.float32)

                # 侧面
                obj_tag = np.array([
                    [-half_size, half_size, 0],
                    [ -half_size,  -half_size, 0],
                    [ half_size, -half_size, 0],
                    [half_size, half_size, 0],
                ], dtype=np.float32)
                success_tag, rvec_tag, tvec_tag = cv2.solvePnP(
                    obj_tag, corners_2d, camera_matrix, dist_coeffs
                )
                if success_tag:
                    draw_axis(img, camera_matrix, dist_coeffs, rvec_tag, tvec_tag, axis_len=TAG_AXIS_LENGTH)

            # base 坐标轴 PnP
            base_pnp_success = False
            if len(obj_pts) >= 4:
                obj_pts_np = np.array(obj_pts, dtype=np.float32)
                img_pts_np = np.array(img_pts, dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts_np, img_pts_np, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    base_pnp_success = True
                    draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len=AXIS_LENGTH)

                    # 如果正在采样，则记录 rvec/tvec（保存为列向量或扁平）
                    if recording:
                        # 保持统一形状：3,1
                        collected_rvecs.append(rvec.reshape(3, 1).copy())
                        collected_tvecs.append(tvec.reshape(3, 1).copy())

            # 窗口提示信息
            status_text = f"{'RECORDING' if recording else 'IDLE'} - samples: {len(collected_rvecs)}"
            cv2.putText(img, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if recording else (0, 200, 200), 2)
            cv2.putText(img, "Press 'p' start/stop sampling & save, ESC quit", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("base calibration", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                # 切换 recording 状态
                recording = not recording
                if recording:
                    print("[INFO] Started sampling. Move tags stably in view. Collecting r/t when base PnP succeeds.")
                else:
                    # 停止采样时，如果有采样则计算平均并保存
                    if len(collected_rvecs) > 0:
                        try:
                            R_avg, t_avg = average_transform(collected_rvecs, collected_tvecs)
                            rvec_avg, _ = cv2.Rodrigues(R_avg)
                            T_base_to_camera = np.eye(4, dtype=np.float64)
                            T_base_to_camera[:3, :3] = R_avg
                            T_base_to_camera[:3, 3] = t_avg.reshape(3)
                            out_path = save_camera_T(REPO_ID, T_base_to_camera)
                            print(f"[INFO] Averaged {len(collected_rvecs)} samples and saved transform to {out_path}")
                            saved_once = True
                        except Exception as e:
                            print(f"[ERROR] Failed to average/save transforms: {e}")
                    else:
                        print("[WARN] No samples collected to average.")
            elif key == 27:  # ESC
                print("[INFO] ESC pressed. Exiting.")
                break

        # 退出循环后，若存在未保存的数据（曾经采样但未保存），自动保存
        if len(collected_rvecs) > 0 and not saved_once:
            print("[INFO] Found unsaved samples - computing average and saving before exit.")
            try:
                R_avg, t_avg = average_transform(collected_rvecs, collected_tvecs)
                rvec_avg, _ = cv2.Rodrigues(R_avg)
                T_base_to_camera = np.eye(4, dtype=np.float64)
                T_base_to_camera[:3, :3] = R_avg
                T_base_to_camera[:3, 3] = t_avg.reshape(3)
                out_path = save_camera_T(REPO_ID, T_base_to_camera)
                print(f"[INFO] Averaged {len(collected_rvecs)} samples and saved transform to {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed to average/save transforms on exit: {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()