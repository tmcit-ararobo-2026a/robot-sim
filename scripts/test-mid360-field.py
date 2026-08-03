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
usd_context.new_stage()

stage = usd_context.get_stage()

UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)

usd_path = os.path.abspath("data/assets/2026field/2026field.usd")

field_prim = stage.DefinePrim("/World/FieldPrim", "Xform")
field_prim.GetReferences().AddReference(usd_path)

for _ in range(10):
    simulation_app.update()

# ---------------------------------------------------------
# 2. USD メッシュから Warp Mesh を作成 & 更新する機能
# ---------------------------------------------------------
warp_mesh_data = {
    'wp_mesh': None,
    'wp_pts': None,
    'wp_idx': None,
    'mesh_prims': []
}

def create_warp_mesh_from_usd():
    all_vertices = []
    all_indices = []
    index_offset = 0
    mesh_prims = []

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

            tri_indices = []
            curr_idx = 0
            for count in face_counts:
                for i in range(1, count - 1):
                    tri_indices.append(indices_np[curr_idx])
                    tri_indices.append(indices_np[curr_idx + i])
                    tri_indices.append(indices_np[curr_idx + i + 1])
                curr_idx += count

            tri_indices_np = np.array(tri_indices, dtype=np.int32)

            transform = mesh_geom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transform_matrix = np.array(transform, dtype=np.float32)

            pts_homo = np.hstack([pts_np, np.ones((len(pts_np), 1), dtype=np.float32)])
            pts_transformed = (pts_homo @ transform_matrix)[:, :3]

            all_vertices.append(pts_transformed)
            all_indices.append(tri_indices_np + index_offset)
            index_offset += len(pts_transformed)
            
            mesh_prims.append((mesh_geom, pts_homo))

    merged_vertices = np.vstack(all_vertices).astype(np.float32)
    merged_indices = np.concatenate(all_indices).astype(np.int32)

    wp_pts = wp.array(merged_vertices, dtype=wp.vec3, device='cuda:0')
    wp_idx = wp.array(merged_indices, dtype=int, device='cuda:0')

    # 【修正】support_refit キーワード引数を削除
    wp_mesh = wp.Mesh(points=wp_pts, indices=wp_idx)

    warp_mesh_data['wp_mesh'] = wp_mesh
    warp_mesh_data['wp_pts'] = wp_pts
    warp_mesh_data['wp_idx'] = wp_idx
    warp_mesh_data['mesh_prims'] = mesh_prims

    return wp_mesh

def update_warp_mesh_transforms():
    """USD上の物体移動に合わせて既存の Warp Mesh 頂点を更新し refit() する"""
    all_transformed_pts = []
    
    timeline = omni.timeline.get_timeline_interface()
    current_time_sec = timeline.get_current_time()
    fps = stage.GetTimeCodesPerSecond()
    time_code = Usd.TimeCode(current_time_sec * fps)

    for mesh_geom, cached_pts_homo in warp_mesh_data['mesh_prims']:
        points = mesh_geom.GetPointsAttr().Get(time_code)
        if points is None or len(points) == 0:
            points = mesh_geom.GetPointsAttr().Get(Usd.TimeCode.Default())
        
        if points is not None and len(points) > 0:
            pts_np = np.array(points, dtype=np.float32)
            pts_homo = np.hstack([pts_np, np.ones((len(pts_np), 1), dtype=np.float32)])
        else:
            pts_homo = cached_pts_homo

        transform = mesh_geom.ComputeLocalToWorldTransform(time_code)
        transform_matrix = np.array(transform, dtype=np.float32)

        pts_transformed = (pts_homo @ transform_matrix)[:, :3]
        all_transformed_pts.append(pts_transformed)

    if not all_transformed_pts:
        return

    updated_vertices = np.vstack(all_transformed_pts).astype(np.float32)

    # 既存の wp.array に書き込み後、refit() を呼ぶ
    new_pts_wp = wp.array(updated_vertices, dtype=wp.vec3, device='cuda:0')
    wp.copy(warp_mesh_data['wp_pts'], new_pts_wp)
    warp_mesh_data['wp_mesh'].refit()

# メッシュの初期化
field_warp_mesh = create_warp_mesh_from_usd()
print(f"[INFO] Field Warp Mesh created. ID: {field_warp_mesh.id}")

# ---------------------------------------------------------
# 3. LidarSensor のセットアップ & 描画ループ
# ---------------------------------------------------------
num_envs = 1
num_sensors = 1

mesh_ids_array = wp.array([field_warp_mesh.id], dtype=wp.uint64, device='cuda:0')

sensor_pos = torch.tensor([[1.0, 0.0, 0.5]], device='cuda:0')
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

# メインループ
for i in range(2000):
    simulation_app.update()

    # 1. Mesh 頂点を更新して refit()（Mesh ID は変えず BVH のみ更新）
    update_warp_mesh_transforms()

    # 2. LiDAR Raycast 実行
    points_tensor, dist_tensor = lidar.update()
    
    pts = points_tensor.detach().cpu().numpy().reshape(-1, 3)

    valid_mask = ~np.isnan(pts).any(axis=1) & (np.linalg.norm(pts, axis=1) > 0.01)
    pts_valid = pts[valid_mask]

    # デバッグ出力（Raw Lidar Output Mean が Mesh Center と連動するか確認）
    if len(pts_valid) > 0 and i % 50 == 0:
        current_wp_pts = warp_mesh_data['wp_pts'].numpy()
        mesh_center = current_wp_pts.mean(axis=0)
        raw_mean = pts_valid.mean(axis=0)
        print(f"[Frame {i:04d}] Mesh Center (World): {mesh_center} | Raw Lidar Output Mean: {raw_mean}")

    # 描画処理
    if len(pts_valid) > 0:
        current_sensor_pos = dummy_env['sensor_pos_tensor'][0].cpu().numpy()
        draw_pts = pts_valid + current_sensor_pos

        draw.clear_points()
        points_list = [tuple(p) for p in draw_pts]
        sizes = [3.0] * len(points_list)
        colors = [(1.0, 0.0, 0.0, 1.0)] * len(points_list)
        
        draw.draw_points(points_list, colors, sizes)