# 更新日志
1. 2025-12-22基础代码上传，并建立新的文件系统(修改了pid)
2. 2025-12-23修改ee部分的采集代码，将data保存为ee-action和joint-action这两种形式，修改了PID以增加稳定性。新增tool/rename_feature。验证了ee的采集、保存以及转换。新增so100模型
3. 2025-12-25增加相机相对base坐标系的标定以及新的数据生成
4. 2025-12-30 ACT+不同位置相机的训练、验证代码,SmolVLA修改了但还是没有执行。
5. 2026/1/12 增加深度相机的保存+不同位置的采集


# 采集所有数据
1. 硬件
   1. sudo chmod 666 /dev/ttyACM0
   2. sudo chmod 666 /dev/ttyACM1
   3. 相机位置
2. 修改`examples/so100_to_so100_EE/record.py`里面的repo_id和episode设置
3. python examples/so100_to_so100_EE/record.py
4. 采集好以后获得相机位置
   1. 标定纸放好
   2. personal/tools/local/camera_calib/detect_average_save.py
5. 如果多录制了怎么删掉之前的lerobot-edit-dataset \
    --repo_id mcamera/second \
    --operation.type delete_episodes \
    --operation.episode_indices "[14]"


# 对采集好的数据处理
1. 相机位置改变+保存ee 修改personal/tools/general/reanme_feature.py，多个数据集也可以直接rename_more_feature.py
2. 合并多个数据集
lerobot-edit-dataset \
    --repo_id lerobot/pusht_merged \
    --operation.type merge \
    --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"
3. 上传到服务器