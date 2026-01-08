import json
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import cv2
from pupil_apriltags import Detector

# ================== 用户参数 ==================
TAG_SIZE = 60.0  # mm
# base坐标系下的位置
TAG_3D_POINTS = {
    0: [128, 163, 0],
    1: [218, 163, 0],
    2: [308, 163, 0],
    3: [128, 73, 0],
    4: [218, 73, 0],
    5: [308, 73, 0],
}
AXIS_LENGTH = 180         # base 坐标轴长度
TAG_AXIS_LENGTH = 30     # 每个 tag 坐标轴长度
# ==============================================

# 画原点，x/y/z轴
# 输入参数：图像（相机），相机内参用于将3d点投影到平面
# dist_coeffs相机的畸变参数，哪来的？
# rvec旋转：坐标系从3d物体到相机的旋转
# tvec平移：坐标系从 3D 物体到相机的平移
def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len=AXIS_LENGTH):
    """画 base 坐标轴或 tag 坐标轴"""
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ])
    # 投影到图片里面
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    o, x, y, z = imgpts.reshape(-1, 2).astype(int)

    cv2.line(img, o, x, (0, 0, 255), 2)   # X 红
    cv2.line(img, o, y, (0, 255, 0), 2)   # Y 绿
    cv2.line(img, o, z, (255, 0, 0), 2)   # Z 蓝
    cv2.circle(img, o, 3, (0, 0, 0), -1)

# 保存 T_base_to_camera 到指定 repo_id 的 meta 目录
def save_camera_T(repo_id: str, T_base_to_camera: np.ndarray):
    HF_LEROBOT_HOME = Path("/home/qwe/.cache/huggingface/lerobot")
    meta_dir = HF_LEROBOT_HOME / repo_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "camera_T.json"

    T_list = T_base_to_camera.tolist()
    with open(out_path, "w") as f:
        json.dump({"T_base_to_camera": T_list}, f, indent=4)

    print(f"[INFO] Saved T_base_to_camera to {out_path}")

def main():
    REPO_ID = input("请输入 dataset repo_id: ").strip()
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
        quad_decimate=1.0, #不缩放
        quad_sigma=0.0,
        refine_edges=1
    )

    print("开始标定，至少看到 4 个 tag 才能解算")

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

                # 如果 tag 在 base 坐标系定义里，用于 base PnP
                if tag_id in TAG_3D_POINTS:
                    obj_pts.append(TAG_3D_POINTS[tag_id])
                    img_pts.append(det.center)

                # --------- 计算每个 tag 坐标轴（solvePnP 用 tag 四角） ---------
                corners_2d = det.corners.astype(np.float32)  # 4x2
                # 右下, 右上, 左上, 左下
                corners_2d_int = det.corners.astype(int)
                for i, corner in enumerate(corners_2d_int):
                    x, y = corner
                    cv2.circle(img, (x, y), 5, (255, 0, 255), -1)
                    cv2.putText(img, f"{i+1}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                half_size = TAG_SIZE / 2
                # pupil_apriltags的默认顺序：左上, 右上, 右下, 左下（没有旋转的时候）在base坐标系中的3d坐标。
                # 这个给的是tag坐标系，还不是base坐标系。
                obj_tag = np.array([
                    [-half_size, -half_size, 0],
                    [ half_size,  -half_size, 0],
                    [ half_size, half_size, 0],
                    [-half_size, half_size, 0],
                ], dtype=np.float32)
                # 输出rvec_tag：旋转向量，将 tag 局部坐标系旋转到相机坐标系；tvec_tag：平移向量，将 tag 局部坐标系平移到相机坐标系
                success_tag, rvec_tag, tvec_tag = cv2.solvePnP(
                    obj_tag, corners_2d, camera_matrix, dist_coeffs
                )
                if success_tag:
                    # 这里的rvec_tag和tvec_tag是每个tag自己的坐标系。
                    draw_axis(img, camera_matrix, dist_coeffs, rvec_tag, tvec_tag, axis_len=TAG_AXIS_LENGTH)

            # --------- base 坐标轴 PnP ---------
            if len(obj_pts) >= 4:
                obj_pts_np = np.array(obj_pts, dtype=np.float32)
                img_pts_np = np.array(img_pts, dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts_np, img_pts_np, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len=AXIS_LENGTH)

            cv2.imshow("base calibration", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                break
        R, _ = cv2.Rodrigues(rvec)

        T_base_to_camera = np.eye(4)
        T_base_to_camera[:3, :3] = R
        T_base_to_camera[:3, 3] = tvec.reshape(3)
        print(T_base_to_camera)
        save_camera_T(REPO_ID, T_base_to_camera)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


### 使用
# def load_camera_T(repo_id: str) -> np.ndarray:
#     HF_LEROBOT_HOME = Path("/home/qwe/.cache/huggingface/lerobot")
#     path = HF_LEROBOT_HOME / repo_id / "meta" / "camera_T.json"
#     if not path.exists():
#         raise FileNotFoundError(f"Camera transform not found for repo {repo_id} at {path}")
#     with open(path) as f:
#         data = json.load(f)
#     return np.array(data["T_base_to_camera"])