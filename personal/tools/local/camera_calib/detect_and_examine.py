import pyrealsense2 as rs
import numpy as np
import cv2
from pupil_apriltags import Detector

# ================== 用户参数 ==================
TAG_SIZE = 60.0  # mm
TAG_3D_POINTS = {
    0: [129,    163,    0],
    1: [219,  163,    0],
    2: [309,  163,    0],
    3: [129,  73,    0],
    4: [219,73,    0],
    5: [309,73,    0],
}
AXIS_LENGTH = 80  # mm
# ==============================================


def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec):
    axis_3d = np.float32([
        [0, 0, 0],
        [AXIS_LENGTH, 0, 0],
        [0, AXIS_LENGTH, 0],
        [0, 0, AXIS_LENGTH]
    ])
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    o, x, y, z = imgpts.reshape(-1, 2).astype(int)

    cv2.line(img, o, x, (0, 0, 255), 3)   # X 红
    cv2.line(img, o, y, (0, 255, 0), 3)   # Y 绿
    cv2.line(img, o, z, (255, 0, 0), 3)   # Z 蓝
    cv2.circle(img, o, 5, (0, 0, 0), -1)


def main():
    # ---------- Realsense ----------
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

    # ---------- Apriltag ----------
    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
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
                if tag_id in TAG_3D_POINTS:
                    obj_pts.append(TAG_3D_POINTS[tag_id])
                    img_pts.append(det.center)

                    c = tuple(det.center.astype(int))
                    cv2.circle(img, c, 5, (0,255,255), -1)
                    cv2.putText(img, f"id:{tag_id}", c,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            if len(obj_pts) >= 4:
                obj_pts = np.array(obj_pts, dtype=np.float32)
                img_pts = np.array(img_pts, dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec)
                     # ====== 2. 重投影误差验证 ======
                    proj_pts, _ = cv2.projectPoints(
                        obj_pts, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    proj_pts = proj_pts.reshape(-1, 2)

                    # 单点误差（像素）
                    errors = np.linalg.norm(proj_pts - img_pts, axis=1)
                    mean_error = errors.mean()

                    # 打印
                    print(f"Reprojection error: {mean_error:.2f} px")

                    # 在图像上画出投影点（红色）
                    for p in proj_pts.astype(int):
                        cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)

            cv2.imshow("base calibration", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
       