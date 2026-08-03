# SimulationApp 起動やパス追加、インポートは既存のまま
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
lidar_path = os.path.expanduser("~/ws/OmniPerception/LidarSensor")
if lidar_path not in sys.path:
    sys.path.append(lidar_path)

from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

import numpy as np
import torch
import warp as wp
import omni.usd
from pxr import Usd, UsdGeom, UsdLux
import omni.timeline
from isaacsim.util.debug_draw import _debug_draw

# ---------------------------------------------------------
# 1. USD ステージを開く / field.usd をロード
# ---------------------------------------------------------
usd_context = omni.usd.get_context()
usd_context.new_stage()  # 新しい空ステージを作成（デフォルトで Z-Up）

stage = usd_context.get_stage()

# ステージの UpAxis を明確に Z-Up に設定
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # 単位をmに統一

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)  # 明るさを設定

# field.usd の絶対パス
usd_path = os.path.abspath("data/assets/2026field/2026field.usd")

# /World/FieldPrim として field.usd を Reference 読み込み
field_prim = stage.DefinePrim("/World/FieldPrim", "Xform")
field_prim.GetReferences().AddReference(usd_path)

# もし field.usd 自体が Y-Up で作られている場合、X軸周りに 90 度回転させて Z-Up に起こす
# (pxr の UsdGeomXformCommonAPI を利用)
xform_api = UsdGeom.XformCommonAPI(field_prim)
# xform_api.SetRotate((0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
# 
# field_scale = 1.0
# xform_api.SetScale((field_scale, field_scale, field_scale))

# 参照の非同期ロードを確実に完了させるため数ステップ更新
for _ in range(10):
    simulation_app.update()

import omni.kit.viewport.utility as vp_utils
from pxr import UsdGeom, Gf

# # 1. カスタムカメラの作成
# camera_path = "/World/TestCamera"
# camera_prim = UsdGeom.Camera.Define(stage, camera_path)
# 
# # カメラの位置（Eye）と向き（LookAt: 原点方向）を設定
# # X=0, Y=-14m, Z=10m から斜め下を見下ろす設定
# xform_api = UsdGeom.XformCommonAPI(camera_prim)
# xform_api.SetTranslate((10.0, -14.0, 12.0))
# xform_api.SetRotate((25.0, 0.0, 30.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
# 
# # 2. Viewport のアクティブカメラに指定
# active_viewport = vp_utils.get_active_viewport()
# if active_viewport:
#     active_viewport.camera_path = camera_path

# ---------------------------------------------------------
# 2. USD メッシュから Warp Mesh を作成する関数 (行列適用・正確版)
# ---------------------------------------------------------
def create_warp_mesh_from_usd():
    all_vertices = []
    all_indices = []
    index_offset = 0

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_geom = UsdGeom.Mesh(prim)
            
            points = mesh_geom.GetPointsAttr().Get()
            face_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
            face_counts = mesh_geom.GetFaceVertexCountsAttr().Get()

            if points is None or face_indices is None:
                continue

            pts_np = np.array(points, dtype=np.float32)
            indices_np = np.array(face_indices, dtype=np.int32)

            if len(pts_np) == 0 or len(indices_np) == 0:
                continue

            # N角形ポリゴンを三角形に分割
            tri_indices = []
            curr_idx = 0
            for count in face_counts:
                for i in range(1, count - 1):
                    tri_indices.append(indices_np[curr_idx])
                    tri_indices.append(indices_np[curr_idx + i])
                    tri_indices.append(indices_np[curr_idx + i + 1])
                curr_idx += count

            tri_indices_np = np.array(tri_indices, dtype=np.int32)

            # --- ワールド変換行列の正確な適用 ---
            # USD の Gf.Matrix4d は row-major なので (N, 4) @ (4, 4) でそのままワールド座標になる
            transform = mesh_geom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transform_matrix = np.array(transform, dtype=np.float32) # (4, 4)

            pts_homo = np.hstack([pts_np, np.ones((len(pts_np), 1), dtype=np.float32)])
            # (N, 4) @ (4, 4) で各頂点にワールド変換（回転・スケール・平行移動）を正しく適用
            pts_transformed = (pts_homo @ transform_matrix)[:, :3]

            all_vertices.append(pts_transformed)
            all_indices.append(tri_indices_np + index_offset)
            index_offset += len(pts_transformed)

    if not all_vertices:
        raise RuntimeError("No valid UsdGeom.Mesh found in the stage!")

    merged_vertices = np.vstack(all_vertices).astype(np.float32)
    merged_indices = np.concatenate(all_indices).astype(np.int32)

    # Bounds 出力による検証
    x_min, y_min, z_min = merged_vertices.min(axis=0)
    x_max, y_max, z_max = merged_vertices.max(axis=0)
    
    print("=" * 60)
    print(f"[DEBUG] Final Synchronized Bounds:")
    print(f"        X: {x_min:.3f} m ~ {x_max:.3f} m (Width: {x_max - x_min:.3f} m)")
    print(f"        Y: {y_min:.3f} m ~ {y_max:.3f} m (Depth: {y_max - y_min:.3f} m)")
    print(f"        Z: {z_min:.3f} m ~ {z_max:.3f} m (Height: {z_max - z_min:.3f} m)")
    print("=" * 60)

    # Warp Mesh の構築
    wp_pts = wp.array(merged_vertices, dtype=wp.vec3, device='cuda:0')
    wp_idx = wp.array(merged_indices, dtype=int, device='cuda:0')

    return wp.Mesh(points=wp_pts, indices=wp_idx)

# メッシュの初期化
field_warp_mesh = create_warp_mesh_from_usd()
print(f"[INFO] Field Warp Mesh created. ID: {field_warp_mesh.id}")

# ---------------------------------------------------------
# 3. LidarSensor のセットアップ & 描画ループ修正
# ---------------------------------------------------------
num_envs = 1
num_sensors = 1

mesh_ids_array = wp.array([field_warp_mesh.id], dtype=wp.uint64, device='cuda:0')

# センサー位置を X=1.0, Y=0.0, Z=0.5m など分かりやすい高さに設定
sensor_pos = torch.tensor([[1.0, 0.0, 0.5]], device='cuda:0')
# クウォータニオン: [x, y, z, w] -> 単位クウォータニオン (回転なし)
sensor_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device='cuda:0')

dummy_env = {
    'num_envs': num_envs,
    'mesh_ids': mesh_ids_array,
    'sensor_pos_tensor': sensor_pos,
    'sensor_quat_tensor': sensor_quat,
}

lidar_config = LidarConfig(
    sensor_type=LidarType.MID360,
    max_range=30.0,
    enable_sensor_noise=False
)

lidar = LidarSensor(
    env=dummy_env, 
    env_cfg={}, 
    sensor_config=lidar_config, 
    num_sensors=num_sensors, 
    device='cuda:0'
)

timeline = omni.timeline.get_timeline_interface()
timeline.play()

draw = _debug_draw.acquire_debug_draw_interface()
sensor_pos_np = sensor_pos[0].cpu().numpy()

for i in range(2000):
    simulation_app.update()
    
    points_tensor, dist_tensor = lidar.update()
    
    # テンソルを numpy 配列へ変換
    pts = points_tensor.detach().cpu().numpy().reshape(-1, 3)

    # 有効点抽出
    valid_mask = ~np.isnan(pts).any(axis=1) & (np.linalg.norm(pts, axis=1) > 0.01)
    pts_valid = pts[valid_mask]

    if len(pts_valid) > 0:
        if i == 0:
            print(f"[DEBUG] Total Valid Points: {len(pts_valid)}")
            print(f"[DEBUG] Raw Point 0: {pts_valid[0]}")

        # --- 重要: 判定の条件分岐を排除 ---
        # LidarSensor クラスは「センサー原点のローカル座標」で点群を返す仕様です。
        # そのため、ワールドに描画する場合は一律で sensor_pos を加算します。
        draw_pts = pts_valid + sensor_pos_np

        draw.clear_points()
        points_list = [tuple(p) for p in draw_pts]
        sizes = [3.0] * len(points_list)
        colors = [(1.0, 0.0, 0.0, 1.0)] * len(points_list)  # 赤
        
        draw.draw_points(points_list, colors, sizes)