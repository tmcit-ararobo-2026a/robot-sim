import os
import pathlib
import sys
import numpy as np
import torch
import warp as wp

# SimulationApp をインポート・初期化する前に環境変数を完全に設定する
isaac_sim_path = pathlib.Path.home() / "isaac-sim-standalone-5.1.0-linux-x86_64"

from isaacsim import SimulationApp

streaming_kit = isaac_sim_path / "apps" / "isaacsim.exp.full.streaming.kit"

simulation_app = SimulationApp({
    "headless": False,
})

import omni.kit.app
app = omni.kit.app.get_app()
# または、エクステンションマネージャー経由で isaacsim.ros2.bridge を有効化
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

import omni.usd
import omni.timeline
from pxr import Usd, UsdGeom, UsdLux
from isaacsim.util.debug_draw import _debug_draw

# --- ROS 2 関連のインポート ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


def create_point_cloud2(points: np.ndarray, frame_id: str = "lidar_link", stamp=None) -> PointCloud2:
    """numpy の (N, 3) float32 配列を sensor_msgs/PointCloud2 に変換するヘルパー"""
    msg = PointCloud2()
    
    header = Header()
    if stamp is not None:
        header.stamp = stamp
    else:
        header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id
    msg.header = header

    if len(points.shape) == 2:
        msg.height = 1
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = 0

    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    
    msg.is_bigendian = False
    msg.point_step = 12  # 3 fields * 4 bytes (float32)
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    
    return msg


class WarpUSDManager:
    """USD上の全Meshを結合し、Warp Meshの生成およびリアルタイムトランスフォーム追従(refit)を管理するクラス"""

    def __init__(self, stage: Usd.Stage, device: str = "cuda:0"):
        self.stage = stage
        self.device = device

        self.mesh_prims = []
        self.wp_pts = None
        self.wp_idx = None
        self.wp_mesh = None

        self._build_warp_mesh()

    def _build_warp_mesh(self):
        all_vertices = []
        all_indices = []
        index_offset = 0

        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

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

            self.mesh_prims.append((mesh_geom, pts_homo))

        if not all_vertices:
            raise RuntimeError("USD ステージ上に有効な Mesh ジオメトリが見つかりませんでした。")

        merged_vertices = np.vstack(all_vertices).astype(np.float32)
        merged_indices = np.concatenate(all_indices).astype(np.int32)

        self.wp_pts = wp.array(merged_vertices, dtype=wp.vec3, device=self.device)
        self.wp_idx = wp.array(merged_indices, dtype=int, device=self.device)

        self.wp_mesh = wp.Mesh(points=self.wp_pts, indices=self.wp_idx)

    def update_transforms(self):
        all_transformed_pts = []

        timeline = omni.timeline.get_timeline_interface()
        current_time_sec = timeline.get_current_time()
        fps = self.stage.GetTimeCodesPerSecond()
        time_code = Usd.TimeCode(current_time_sec * fps)

        for mesh_geom, cached_pts_homo in self.mesh_prims:
            points = mesh_geom.GetPointsAttr().Get(time_code)
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

        wp.copy(self.wp_pts, wp.array(updated_vertices, dtype=wp.vec3, device=self.device))
        self.wp_mesh.refit()


def main():
    # --- ROS 2 ノードの初期化 ---
    rclpy.init()
    ros_node = rclpy.create_node("isaac_mid360_publisher")
    lidar_pub = ros_node.create_publisher(PointCloud2, "/lidar/points", 10)

    device = "cuda:0"

    # 1. USD ステージの準備
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

    # 2. Warp Mesh 管理オブジェクトの生成
    warp_mgr = WarpUSDManager(stage, device=device)
    print(f"[INFO] Field Warp Mesh Created. Mesh ID: {warp_mgr.wp_mesh.id}")

    # 3. LidarSensor の構築
    mesh_ids_array = wp.array([warp_mgr.wp_mesh.id], dtype=wp.uint64, device=device)
    sensor_pos = torch.tensor([[1.0, 0.0, 0.5]], device=device)
    sensor_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device)

    dummy_env = {
        "num_envs": 1,
        "mesh_ids": mesh_ids_array,
        "sensor_pos_tensor": sensor_pos,
        "sensor_quat_tensor": sensor_quat,
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
        num_sensors=1,
        device=device
    )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    draw = _debug_draw.acquire_debug_draw_interface()

    # 4. メインループ
    try:
        frame_idx = 0
        while simulation_app.is_running():
            simulation_app.update()

            # ROS 2 コールバックの簡易スピン処理
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            # Dynamic Mesh 更新 (refit)
            warp_mgr.update_transforms()

            # Raycast 計測
            points_tensor, _ = lidar.update()
            pts = points_tensor.detach().cpu().numpy().reshape(-1, 3)

            # 有効点フィルター
            valid_mask = ~np.isnan(pts).any(axis=1) & (np.linalg.norm(pts, axis=1) > 0.01)
            pts_valid = pts[valid_mask]

            # ROS 2 点群配信
            if len(pts_valid) > 0:
                current_sensor_pos = dummy_env["sensor_pos_tensor"][0].cpu().numpy()
                # センサー座標系の点群をワールド座標（またはそのまま）にする場合は調整
                publish_pts = pts_valid + current_sensor_pos

                pc_msg = create_point_cloud2(publish_pts, frame_id="lidar_link")
                lidar_pub.publish(pc_msg)

            # デバッグログ出力
            if len(pts_valid) > 0 and frame_idx % 50 == 0:
                mesh_center = warp_mgr.wp_pts.numpy().mean(axis=0)
                raw_mean = pts_valid.mean(axis=0)
                print(f"[Frame {frame_idx:04d}] Mesh Center: {mesh_center} | Raw Lidar Output Mean: {raw_mean}")

            # 点群の描画
            if len(pts_valid) > 0:
                current_sensor_pos = dummy_env["sensor_pos_tensor"][0].cpu().numpy()
                draw_pts = pts_valid + current_sensor_pos

                draw.clear_points()
                points_list = [tuple(p) for p in draw_pts]
                draw.draw_points(points_list, [(1.0, 0.0, 0.0, 1.0)] * len(points_list), [3.0] * len(points_list))

            frame_idx += 1

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()