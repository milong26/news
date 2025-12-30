import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# ================== 参数 ==================
TAG_SIZE = 0.04   # 40mm
TAG_ID = 1
AVG_N = 100

# ================== RealSense 初始化 ==================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# ===== 自动获取彩色相机内参 =====
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy
camera_params = (fx, fy, cx, cy)

print("使用 RealSense 彩色相机内参:")
print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

# ================== AprilTag Detector ==================
detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# ================== 数据缓存 ==================
samples = []

print("AprilTag 平均位姿检测开始（ESC 退出）")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE
        )

        for det in detections:
            if det.tag_id != TAG_ID:
                continue

            t = det.pose_t.flatten()
            samples.append(t)

            # 画 tag
            corners = det.corners.astype(int)
            cv2.polylines(color_img, [corners], True, (0, 255, 0), 2)

            if len(samples) >= AVG_N:
                avg = np.mean(samples, axis=0)
                std = np.std(samples, axis=0)

                print(
                    f"[AVG {AVG_N}] "
                    f"tx={avg[0]:.4f}, "
                    f"ty={avg[1]:.4f}, "
                    f"tz={avg[2]:.4f} | "
                    f"std(mm)=({std[0]*1000:.2f}, "
                    f"{std[1]*1000:.2f}, "
                    f"{std[2]*1000:.2f})"
                )

                samples.clear()

        cv2.imshow("AprilTag Detection", color_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
