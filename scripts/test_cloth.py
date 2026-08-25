# ClothLauncher Script Node (完全状態リセット・重複発射防止版)
import numpy as np
from collections import deque
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.prims import is_prim_path_valid
import omni.usd

def _is_valid_vec(v):
    return np.all(np.isfinite(v))

def _is_valid_quat(q):
    n = np.linalg.norm(q)
    return np.isfinite(n) and n > 1e-6

def _quat_rotate(quat_wxyz, vec):
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (uv * w + uuv)

def setup(db):
    state = db.internal_state
    stage = omni.usd.get_context().get_stage()
    state.stage = stage
    
    # シミュレーション開始時に状態を完全に初期化（過去の送信記録を消去）
    state.has_fired_for_current_msg = True  # Play直後の誤発火を防止
    state.last_raw_vel = None

    pool_root = db.inputs.poolRootPrim
    if not pool_root:
        db.log_error("[ClothLauncher] poolRootPrim is empty")
        return True
    pool_root_path = str(pool_root[0])
    pool_size = int(db.inputs.poolSize)

    state.available = deque()

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

    local_lin_vel = np.array([db.inputs.linearVelocity[0], db.inputs.linearVelocity[1], db.inputs.linearVelocity[2]], dtype=np.float64)
    local_ang_vel = np.array([db.inputs.angularVelocity[0], db.inputs.angularVelocity[1], db.inputs.angularVelocity[2]], dtype=np.float64)

    current_vel_tuple = (round(local_lin_vel[0], 3), round(local_lin_vel[1], 3), round(local_lin_vel[2], 3))

    # 速度がほぼ 0 の場合はリセットして終了
    if np.linalg.norm(local_lin_vel) < 1e-3 and np.linalg.norm(local_ang_vel) < 1e-3:
        state.has_fired_for_current_msg = False
        state.last_raw_vel = current_vel_tuple
        return True

    # 同じメッセージがバッファに残っている場合のチェック
    if state.last_raw_vel == current_vel_tuple:
        if state.has_fired_for_current_msg:
            return True
    else:
        # 新しい値が届いたときのみ新規発射を許可
        state.last_raw_vel = current_vel_tuple
        state.has_fired_for_current_msg = False

    # プール残数チェック
    if not state.available:
        db.log_warning(f"[ClothLauncher] pool exhausted (remaining: {len(state.available)})")
        return True

    # 発射基準点（ReleasePoint）の取得
    release_prim_target = db.inputs.releasePointPrim
    if not release_prim_target:
        db.log_error("[ClothLauncher] releasePointPrim is empty")
        return True
    
    release_prim = stage.GetPrimAtPath(str(release_prim_target[0]))
    if not release_prim:
        db.log_error(f"[ClothLauncher] releasePointPrim not found: {release_prim_target[0]}")
        return True

    # 発射位置と姿勢の計算
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mat = xform_cache.GetLocalToWorldTransform(release_prim)
    world_mat.Orthonormalize()
    world_pos = world_mat.ExtractTranslation()
    world_quat = world_mat.ExtractRotationQuat().GetNormalized()

    world_pos_np = np.array([world_pos[0], world_pos[1], world_pos[2]], dtype=np.float64)
    quat_wxyz = np.array([
        world_quat.GetReal(),
        world_quat.GetImaginary()[0],
        world_quat.GetImaginary()[1],
        world_quat.GetImaginary()[2],
    ], dtype=np.float64)

    if not (_is_valid_vec(world_pos_np) and _is_valid_quat(quat_wxyz)):
        db.log_warning(f"[ClothLauncher] invalid release pose, skip")
        return True
    quat_wxyz /= np.linalg.norm(quat_wxyz)

    # 速度の変換（ローカル -> ワールド）
    world_lin_vel = _quat_rotate(quat_wxyz, local_lin_vel)
    world_ang_vel = _quat_rotate(quat_wxyz, local_ang_vel)

    if not (_is_valid_vec(world_lin_vel) and _is_valid_vec(world_ang_vel)):
        db.log_warning(f"[ClothLauncher] invalid velocity, skip")
        return True

    # プールから1つ取り出し
    marker_path = state.available.popleft()
    marker_prim = stage.GetPrimAtPath(marker_path)

    # USD Xformable で位置移動
    xformable = UsdGeom.Xformable(marker_prim)
    xformable.ClearXformOpOrder()
    rot_gf = Gf.Rotation(Gf.Quatd(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
    transform_matrix = Gf.Matrix4d().SetTransform(rot_gf, Gf.Vec3d(*world_pos_np))
    xformable.AddTransformOp().Set(transform_matrix)

    # Cloth（simulation_mesh）の頂点へ速度を適用
    simulation_mesh_path = f"{marker_path}/simulation_mesh"
    simulation_mesh_prim = stage.GetPrimAtPath(simulation_mesh_path)
    if simulation_mesh_prim:
        simulation_mesh = UsdGeom.Mesh(simulation_mesh_prim)
        sim_points = simulation_mesh.GetPointsAttr().Get()
        if sim_points:
            num_points = len(sim_points)
            velocities = np.empty((num_points, 3), dtype=np.float32)
            velocities[:] = world_lin_vel

            if np.linalg.norm(world_ang_vel) > 1e-6:
                points_np = np.asarray([[p[0], p[1], p[2]] for p in sim_points], dtype=np.float64)
                center = np.mean(points_np, axis=0)
                r = points_np - center
                velocities += np.cross(world_ang_vel, r).astype(np.float32)

            sim_velocities_attr = simulation_mesh.GetVelocitiesAttr()
            if sim_velocities_attr.IsValid():
                sim_velocities_attr.Set([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in velocities])

    UsdGeom.Imageable(marker_prim).MakeVisible()

    # 発射完了フラグをセット
    state.has_fired_for_current_msg = True

    db.outputs.spawnedPrimPath = marker_path
    db.outputs.execOut = True
    return True