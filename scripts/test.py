# 1. まず Isaac Sim の SimulationApp を起動 (Warp などの初期化を正しい内蔵パスで行わせる)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# 2. SimulationApp 起動後に自作ライブラリのパスを追加
import sys
import os

lidar_path = os.path.expanduser("~/ws/OmniPerception/LidarSensor")
if lidar_path not in sys.path:
    sys.path.append(lidar_path)  # insert(0) ではなく append を使用

# 3. ここで自作モジュールをインポート
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

import numpy as np
import warp as wp
import omni.timeline

lidar_config = LidarConfig(
    sensor_type=LidarType.MID360,
    max_range=30.0,
    enable_sensor_noise=False
)

import torch

# LidarSensor が求める env 辞書のダミーを作成
# ※ センサー数・環境数に合わせて shape を指定します
num_envs = 1
num_sensors = 1

dummy_env = {
    'num_envs': num_envs,
    # メッシュID（Warpメッシュ作成済みの場合はその mesh.id、単体テストなら 0 またはダミー値）
    'mesh_ids': 0, 
    # 位置: [num_envs, num_sensors, 3] または [num_envs, 3] (クラスの受け取り方に依存)
    'sensor_pos_tensor': torch.zeros((num_envs, 3), device='cuda:0'),
    # 姿勢 (Quaternion: x, y, z, w または w, x, y, z)
    'sensor_quat_tensor': torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cuda:0'),
}

dummy_env_cfg = {}

# インスタンス化
lidar = LidarSensor(
    env=dummy_env, 
    env_cfg=dummy_env_cfg, 
    sensor_config=lidar_config, 
    num_sensors=num_sensors, 
    device='cuda:0'
)

# タイムラインの取得（Worldの代わり）
timeline = omni.timeline.get_timeline_interface()
timeline.play()

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()