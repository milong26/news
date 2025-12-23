# 检测side相机的位置并可按 w 键保存图像

import pyrealsense2 as rs
import numpy as np
import cv2
import os

def main():
    # 创建保存目录
    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)
    save_count = 0

    # 1. 创建 RealSense 管道
    pipeline = rs.pipeline()
    
    # 2. 配置流（这里只启用彩色图像）
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 3. 启动流
    pipeline.start(config)
    
    print("按下 'q' 退出程序")
    print("按下 'w' 捕获一张图并保存")

    try:
        while True:
            # 4. 等待新的帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # 5. 转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 6. 显示图像
            cv2.imshow('RealSense Color Stream', color_image)
            
            key = cv2.waitKey(1) & 0xFF

            # 按 w 保存
            if key == ord('w'):
                filename = os.path.join(save_dir, f"capture_{save_count:03d}.png")
                cv2.imwrite(filename, color_image)
                print(f"已保存: {filename}")
                save_count += 1
            
            # 按 q 退出
            if key == ord('q'):
                break

    finally:
        # 7. 停止流
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
