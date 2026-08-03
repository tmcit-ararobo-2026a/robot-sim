# 1. まず Isaac Sim の SimulationApp を起動 (Warp などの初期化を正しい内蔵パスで行わせる)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

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

# 10m x 10m の簡単な平面メッシュ（4頂点 / 2ポリゴン）を作成
vertices_np = np.array([
    [-5.0, -5.0, -0.5],
    [ 5.0, -5.0, -0.5],
    [ 5.0,  5.0, -0.5],
    [-5.0,  5.0, -0.5]
], dtype=np.float32)

indices_np = np.array([
    0, 1, 2,
    0, 2, 3
], dtype=np.int32)

# Warp 配列化して wp.Mesh オブジェクトを構築
wp_vertices = wp.array(vertices_np, dtype=wp.vec3, device='cuda:0')
wp_indices = wp.array(indices_np, dtype=int, device='cuda:0')
dummy_mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)

print(f"[TEST] Created Dummy Warp Mesh ID: {dummy_mesh.id}")

# dummy_mesh.id (有効な C++ ポインタ値) を Warp 配列にして渡す
mesh_ids_array = wp.array([dummy_mesh.id], dtype=wp.uint64, device='cuda:0')

dummy_env = {
    'num_envs': num_envs,
    'mesh_ids': mesh_ids_array,  # 正しい Mesh ID が入った Warp 配列
    'sensor_pos_tensor': torch.zeros((num_envs, 3), device='cuda:0'),
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

from isaacsim.util.debug_draw import _debug_draw

draw = _debug_draw.acquire_debug_draw_interface()

print(f"num_envs: {lidar.num_envs}")
print(f"num_sensors: {lidar.num_sensors}")
print(f"num_vertical_lines: {lidar.num_vertical_lines}")
print(f"num_horizontal_lines: {lidar.num_horizontal_lines}")
print(f"lidar_tensor shape: {lidar.lidar_tensor.shape}")
print(f"ray_vectors shape: {lidar.ray_vectors.shape if hasattr(lidar, 'ray_vectors') else 'N/A'}")

# ループ内で実行
for i in range(500):
    simulation_app.update() #[cite: 1]
    
    # update() から点群テンソル (pixels_tensor) と距離データ (dist_tensor) を受け取る
    points_tensor, dist_tensor = lidar.update()
    
    draw.clear_points() #[cite: 1]
    
    # テンソルを CPU / NumPy 配列に変換
    points = points_tensor.detach().cpu().numpy().reshape(-1, 3)
    
    # 描画用リストの作成 (点の位置, カラーRGBA, 点のサイズ)
    sizes = [2.0] * len(points) #[cite: 1]
    colors = [(1.0, 0.0, 0.0, 1.0)] * len(points)  # 赤色[cite: 1]
    
    draw.draw_points(points.tolist(), colors, sizes) #[cite: 1]

simulation_app.close()