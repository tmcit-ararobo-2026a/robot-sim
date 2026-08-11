# ClothLauncher Script Node
# Inputs (OmniGraphに手動追加するAttribute):
#   inputs:execIn            (execution)
#   inputs:linearVelocity    (double3)   ← ROS2 Subscribe Twistから接続
#   inputs:angularVelocity   (double3)   ← 同上
#   inputs:releasePointPrim  (target)     デフォルト "/World/Robot/Belt/ReleasePoint"
#   inputs:poolRootPrim      (target)     デフォルト "/World/ClothPool"
#   inputs:poolSize          (int)       デフォルト 32
# Outputs:
#   outputs:execOut          (execution)
#   outputs:spawnedPrimPath  (token)     生成された雑巾のパス（衝突検出ノードへ渡す）

import numpy as np
from collections import deque
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
import omni.usd
import omni.timeline

def _is_valid_vec(v):
    return np.all(np.isfinite(v))


def _is_valid_quat(q):
    n = np.linalg.norm(q)
    return np.isfinite(n) and n > 1e-6

def _quat_rotate(quat_wxyz, vec):
    """四元数(w,x,y,z)でベクトルをワールド座標へ回転"""
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (uv * w + uuv)

def setup(db):
    state = db.internal_state
    stage = omni.usd.get_context().get_stage()
    state.stage = stage
    state.frame_counter = 0

    pool_root = db.inputs.poolRootPrim
    if not pool_root:
        db.log_error("[ClothLauncher] poolRootPrim is empty")
        return True
    pool_root_path = str(pool_root[0])
    pool_size = db.inputs.poolSize

    state.available = deque()
    state.in_flight = {}

    for i in range(pool_size):
        path = f"{pool_root_path}/marker_{i:02d}"
        if not is_prim_path_valid(path):
            db.log_warning(f"[ClothLauncher] marker prim not found: {path}")
            continue
        state.available.append(path)

    db.log_info(f"[ClothLauncher] pool ready: {len(state.available)} markers")

def compute(db):
    state = db.internal_state
    stage = state.stage

    if not state.available:
        db.log_warning("[ClothLauncher] pool exhausted (32 markers used)")
        return True

    marker_path = state.available.popleft()

    release_prim_target = db.inputs.releasePointPrim
    if not release_prim_target:
        db.log_error("[ClothLauncher] releasePointPrim is empty")
        state.available.appendleft(marker_path)
        return True
    release_prim = stage.GetPrimAtPath(str(release_prim_target[0]))
    if not release_prim:
        db.log_error(f"[ClothLauncher] releasePointPrim not found: {release_prim_target[0]}")
        state.available.appendleft(marker_path)
        return True

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mat = xform_cache.GetLocalToWorldTransform(release_prim)
    world_mat.Orthonormalize()                          # ★スケール/せん断除去
    world_pos = world_mat.ExtractTranslation()
    world_quat = world_mat.ExtractRotationQuat().GetNormalized()  # ★正規化

    world_pos_np = np.array([world_pos[0], world_pos[1], world_pos[2]])
    quat_wxyz = np.array([
        world_quat.GetReal(),
        world_quat.GetImaginary()[0],
        world_quat.GetImaginary()[1],
        world_quat.GetImaginary()[2],
    ])

    if not (_is_valid_vec(world_pos_np) and _is_valid_quat(quat_wxyz)):
        db.log_warning(f"[ClothLauncher] invalid release pose, skip: pos={world_pos_np}, quat={quat_wxyz}")
        state.available.appendleft(marker_path)
        return True
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)

    local_lin_vel = np.array([db.inputs.linearVelocity[0], db.inputs.linearVelocity[1], db.inputs.linearVelocity[2]])
    local_ang_vel = np.array([db.inputs.angularVelocity[0], db.inputs.angularVelocity[1], db.inputs.angularVelocity[2]])
    world_lin_vel = _quat_rotate(quat_wxyz, local_lin_vel)
    world_ang_vel = _quat_rotate(quat_wxyz, local_ang_vel)

    if not (_is_valid_vec(world_lin_vel) and _is_valid_vec(world_ang_vel)):
        db.log_warning(f"[ClothLauncher] invalid velocity, skip: lin={world_lin_vel}, ang={world_ang_vel}")
        state.available.appendleft(marker_path)
        return True

    marker = SingleRigidPrim(marker_path)
    marker.set_world_pose(position=world_pos_np, orientation=quat_wxyz)
    marker.set_visibility(True)
    marker.set_linear_velocity(world_lin_vel)
    marker.set_angular_velocity(world_ang_vel)

    state.frame_counter += 1
    state.in_flight[marker_path] = {"released_frame": state.frame_counter}

    db.outputs.spawnedPrimPath = marker_path
    db.outputs.execOut = True
    return True