# ClothLauncher Script Node (World直下・起動時完全非表示対応版)
# ClothLauncher Script Node (実機ベルト直動式射出機構・実測LUT対応版)
import numpy as np
import time
from collections import deque
from pxr import Usd, UsdGeom, Gf, Vt, Sdf
from isaacsim.core.utils.prims import is_prim_path_valid
import omni.usd
import omni.graph.core as og

# ==============================================================================
# Kamagata0/shooter_calculator 実機実測データベース (LUT)
# ベルト直動式射出機構の出力レベル(1〜10)・ベルト速度指令(0.1〜1.0 m/s) ↔ release velocity [m/s]
# ==============================================================================
MEASURED_LUT = [
    # (level, speed_ratio, release_velocity, flight_distance)
    (1,  0.1, 1.550000000, 0.000000000),
    (2,  0.2, 2.296666667, 0.533333333),
    (3,  0.3, 4.763333333, 2.666666667),
    (4,  0.4, 5.953333333, 3.966666667),
    (5,  0.5, 8.840000000, 3.566666667),
    (6,  0.6, 11.23333333, 4.100000000),
    (7,  0.7, 8.516666667, 3.333333333),
    (8,  0.8, 9.040000000, 3.566666667),
    (9,  0.9, 9.420000000, 3.833333333),
    (10, 1.0, 10.84666667, 3.966666667),
]

DEFAULT_LAUNCH_ANGLE_DEG = 50.0  # 実機シューター射出仰角 [deg]

def get_release_velocity_from_level(level):
    """出力レベル (1.0〜10.0) から線形補間で release velocity [m/s] と推定飛距離 [m] を取得"""
    if level <= MEASURED_LUT[0][0]:
        return MEASURED_LUT[0][2], MEASURED_LUT[0][3]
    if level >= MEASURED_LUT[-1][0]:
        return MEASURED_LUT[-1][2], MEASURED_LUT[-1][3]
    
    for i in range(len(MEASURED_LUT) - 1):
        l1, _, v1, d1 = MEASURED_LUT[i]
        l2, _, v2, d2 = MEASURED_LUT[i + 1]
        if l1 <= level <= l2:
            t = (level - l1) / (l2 - l1)
            return v1 + t * (v2 - v1), d1 + t * (d2 - d1)
    return MEASURED_LUT[-1][2], MEASURED_LUT[-1][3]

def get_release_velocity_from_speed_ratio(ratio):
    """/belt/speed_ratio (0.1〜1.0 m/s) から線形補間で release velocity [m/s] を取得"""
    level = ratio * 10.0
    return get_release_velocity_from_level(level)

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

def _extract_prim_path_string(input_val):
    if not input_val:
        return ""
    if isinstance(input_val, (list, tuple, np.ndarray)):
        if len(input_val) == 0:
            return ""
        val = input_val[0]
    else:
        val = input_val
    
    if isinstance(val, Sdf.Path):
        return val.pathString
    return str(val)

def _reset_single_marker(marker_path, stage, init_transform, init_points):
    """プール保管時：物理・コライダーをOFFにし、完全に非表示化＆速度・形状リセット"""
    if not is_prim_path_valid(marker_path):
        return
    
    marker_prim = stage.GetPrimAtPath(marker_path)
    
    # 1. 物理（Deformable Body）を無効化
    if marker_prim.HasAPI("PhysxDeformableBodyAPI"):
        attr = marker_prim.GetAttribute("physxDeformable:deformableBodyEnabled")
        if attr:
            attr.Set(False)
    
    # 2. 完全に非表示にする
    imageable = UsdGeom.Imageable(marker_prim)
    if imageable:
        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    
    # 3. 座標を初期位置に戻す
    xformable = UsdGeom.Xformable(marker_prim)
    xformable.SetResetXformStack(True)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(init_transform)

    # 4. メッシュ形状と速度をリセット
    simulation_mesh_path = f"{marker_path}/simulation_mesh"
    simulation_mesh_prim = stage.GetPrimAtPath(simulation_mesh_path)
    if simulation_mesh_prim:
        simulation_mesh = UsdGeom.Mesh(simulation_mesh_prim)
        
        if marker_path in init_points and init_points[marker_path]:
            simulation_mesh.GetPointsAttr().Set(init_points[marker_path])

        sim_points = simulation_mesh.GetPointsAttr().Get()
        if sim_points:
            num_points = len(sim_points)
            zero_vels = Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * num_points)
            simulation_mesh.GetVelocitiesAttr().Set(zero_vels)

def _reset_all_markers(state, stage):
    state.available = deque()
    for path, init_transform in state.initial_transforms.items():
        if not is_prim_path_valid(path):
            continue
        _reset_single_marker(path, stage, init_transform, state.initial_points)
        state.available.append(path)

def setup(db):
    state = db.internal_state
    stage = omni.usd.get_context().get_stage()
    state.stage = stage
    state.initial_transforms = {}
    state.initial_points = {}
    state.last_launch_time = 0.0

    try:
        node = db.node
        if not node.get_attribute("inputs:reset").is_valid():
            node.create_attribute(
                "inputs:reset",
                og.Type(og.BaseDataType.BOOL),
                custom_user_data="bool input for resetting all cloths",
                port_type=og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
            )
    except Exception:
        pass
    # OmniGraph属性の安全な作成（存在しない場合のみ作成）
    def _ensure_attr(attr_name, base_type, custom_data=""):
        try:
            node = db.node
            if not node.get_attribute(attr_name).is_valid():
                node.create_attribute(
                    attr_name,
                    og.Type(base_type),
                    custom_user_data=custom_data,
                    port_type=og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
                )
        except Exception:
            pass

    _ensure_attr("inputs:reset", og.BaseDataType.BOOL, "bool input for resetting all cloths")
    _ensure_attr("inputs:outputLevel", og.BaseDataType.FLOAT, "Shooter output level (1.0 to 10.0)")
    _ensure_attr("inputs:speedRatio", og.BaseDataType.FLOAT, "Belt speed command ratio (0.1 to 1.0 m/s)")
    _ensure_attr("inputs:launchAngleDeg", og.BaseDataType.FLOAT, "Shooter launch angle in degrees (default 50.0)")
    _ensure_attr("inputs:cooldownTime", og.BaseDataType.FLOAT, "Cooldown interval between launches in seconds")
    _ensure_attr("inputs:releaseVelocity", og.BaseDataType.FLOAT, "Direct release velocity override in m/s")

    # --- /World 直下から直接 marker_00 ~ を探索する設定 ---
    state.root_path = "/World"
    state.pool_size = int(db.inputs.poolSize)

    for i in range(state.pool_size):
        path = f"{state.root_path}/marker_{i:02d}"
        if not is_prim_path_valid(path):
            db.log_warning(f"[ClothLauncher] World直下に該当するパスが見つかりません: {path}")
            continue
        
        marker_prim = stage.GetPrimAtPath(path)
        xformable = UsdGeom.Xformable(marker_prim)
        state.initial_transforms[path] = xformable.GetLocalTransformation(Usd.TimeCode.Default())

        sim_mesh_prim = stage.GetPrimAtPath(f"{path}/simulation_mesh")
        if sim_mesh_prim:
            sim_mesh = UsdGeom.Mesh(sim_mesh_prim)
            state.initial_points[path] = sim_mesh.GetPointsAttr().Get()

    # 起動した瞬間にすべての雑巾を非表示・リセット状態でスタンバイ
    _reset_all_markers(state, stage)
    return True

def compute(db):
    state = db.internal_state
    stage = getattr(state, "stage", None)
    if stage is None:
        stage = omni.usd.get_context().get_stage()
        state.stage = stage

    is_reset = False
    if hasattr(db.inputs, "reset"):
        is_reset = bool(db.inputs.reset)

    if is_reset:
        _reset_all_markers(state, stage)
        try:
            reset_attr = db.node.get_attribute("inputs:reset")
            if reset_attr and reset_attr.is_valid():
                reset_attr.set(False)
        except Exception as e:
            db.log_warning(f"[ClothLauncher] Could not reset input attribute: {e}")
        return True

    if hasattr(db.inputs, "execIn") and not db.inputs.execIn:
        return True

    # --- クールダウン処理（連射制限） ---
    current_time = time.time()
    cooldown = 0.5
    if hasattr(db.inputs, "cooldownTime"):
        try:
            c_val = float(db.inputs.cooldownTime)
            if c_val > 0.0:
                cooldown = c_val
        except Exception:
            pass

    if (current_time - getattr(state, "last_launch_time", 0.0)) < cooldown:
        return True

    if not hasattr(state, "available") or not state.available:
        db.log_warning("[ClothLauncher] pool exhausted!")
        return True

    release_path_str = _extract_prim_path_string(db.inputs.releasePointPrim)
    if not release_path_str:
        return True

    release_prim = stage.GetPrimAtPath(release_path_str)
    if not release_prim:
        return True

    local_lin_vel = np.array([db.inputs.linearVelocity[0], db.inputs.linearVelocity[1], db.inputs.linearVelocity[2]], dtype=np.float64)
    # --- 実機射出計算 (Kamagata0/shooter_calculator LUT) ---
    release_vel = None
    est_distance = 0.0
    mode_name = "linearVelocity"

    # 1. releaseVelocity 直接指定
    if hasattr(db.inputs, "releaseVelocity"):
        try:
            r_val = float(db.inputs.releaseVelocity)
            if r_val > 1e-3:
                release_vel = r_val
                mode_name = "direct_release_velocity"
        except Exception:
            pass

    # 2. outputLevel (1〜10) による実測LUT参照
    if release_vel is None and hasattr(db.inputs, "outputLevel"):
        try:
            lvl = float(db.inputs.outputLevel)
            if lvl >= 1.0:
                release_vel, est_distance = get_release_velocity_from_level(lvl)
                mode_name = f"output_level_{lvl:.1f}"
        except Exception:
            pass

    # 3. speedRatio (0.1〜1.0 m/s) による実測LUT参照
    if release_vel is None and hasattr(db.inputs, "speedRatio"):
        try:
            ratio = float(db.inputs.speedRatio)
            if ratio > 1e-4:
                release_vel, est_distance = get_release_velocity_from_speed_ratio(ratio)
                mode_name = f"speed_ratio_{ratio:.2f}"
        except Exception:
            pass

    # 射出仰角（実機デフォルト 50.0 deg）
    launch_angle_deg = DEFAULT_LAUNCH_ANGLE_DEG
    if hasattr(db.inputs, "launchAngleDeg"):
        try:
            a_val = float(db.inputs.launchAngleDeg)
            if a_val > 0.0:
                launch_angle_deg = a_val
        except Exception:
            pass

    # 従来の linearVelocity の入力確認
    has_custom_lin_vel = False
    raw_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if hasattr(db.inputs, "linearVelocity"):
        try:
            raw_lin_vel = np.array([db.inputs.linearVelocity[0], db.inputs.linearVelocity[1], db.inputs.linearVelocity[2]], dtype=np.float64)
            if np.linalg.norm(raw_lin_vel) > 1e-3:
                has_custom_lin_vel = True
        except Exception:
            pass

    # ローカル初速ベクトルの合成
    rad = np.radians(launch_angle_deg)
    if release_vel is not None:
        if has_custom_lin_vel:
            # 既存の linearVelocity が向きを持っている場合は、その向きに release_vel の大きさを乗算
            dir_vec = raw_lin_vel / np.linalg.norm(raw_lin_vel)
            local_lin_vel = dir_vec * release_vel
        else:
            # 射出口のローカル座標系で仰角 launch_angle_deg の初速ベクトルを作成 (+X前方, +Z上方向)
            local_lin_vel = np.array([np.cos(rad) * release_vel, 0.0, np.sin(rad) * release_vel], dtype=np.float64)
    else:
        if has_custom_lin_vel:
            local_lin_vel = raw_lin_vel
        else:
            # フォールバック：デフォルト出力レベル4（約5.95 m/s, 飛距離約4.0m）
            release_vel, est_distance = get_release_velocity_from_level(4.0)
            local_lin_vel = np.array([np.cos(rad) * release_vel, 0.0, np.sin(rad) * release_vel], dtype=np.float64)
            mode_name = "fallback_level_4"

    marker_path = state.available.popleft()
    marker_prim = stage.GetPrimAtPath(marker_path)

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
        state.available.appendleft(marker_path)
        return True

    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    world_lin_vel = _quat_rotate(quat_wxyz, local_lin_vel)

    xformable = UsdGeom.Xformable(marker_prim)
    xformable.SetResetXformStack(True)
    xformable.ClearXformOpOrder()
    rot_gf = Gf.Rotation(Gf.Quatd(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
    transform_matrix = Gf.Matrix4d().SetTransform(rot_gf, Gf.Vec3d(world_pos_np[0], world_pos_np[1], world_pos_np[2]))
    xformable.AddTransformOp().Set(transform_matrix)

    imageable = UsdGeom.Imageable(marker_prim)
    if imageable:
        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    if marker_prim.HasAPI("PhysxDeformableBodyAPI"):
        attr = marker_prim.GetAttribute("physxDeformable:deformableBodyEnabled")
        if attr:
            attr.Set(True)

    simulation_mesh_path = f"{marker_path}/simulation_mesh"
    simulation_mesh_prim = stage.GetPrimAtPath(simulation_mesh_path)
    if simulation_mesh_prim:
        simulation_mesh = UsdGeom.Mesh(simulation_mesh_prim)
        
        if marker_path in state.initial_points and state.initial_points[marker_path]:
            simulation_mesh.GetPointsAttr().Set(state.initial_points[marker_path])

        sim_points = simulation_mesh.GetPointsAttr().Get()
        if sim_points:
            num_points = len(sim_points)
            velocities = Vt.Vec3fArray([Gf.Vec3f(float(world_lin_vel[0]), float(world_lin_vel[1]), float(world_lin_vel[2]))] * num_points)
            simulation_mesh.GetVelocitiesAttr().Set(velocities)

 # スポーン成功時の処理の最後あたりに追加
    state.last_launch_time = current_time
    vel_mag = np.linalg.norm(local_lin_vel)
    world_pos_list = [float(world_pos_np[0]), float(world_pos_np[1]), float(world_pos_np[2])]
    print(f"[ClothLauncher] Spawned {marker_path} at position: {world_pos_list}")
    print(f"[ClothLauncher] Fired {marker_path}! Mode: {mode_name}, Velocity: {vel_mag:.2f} m/s (Est Dist: {est_distance:.2f}m, Angle: {launch_angle_deg:.1f}deg) at pos: {world_pos_list}")

    print(f"Release Prim Found: {bool(release_prim)}, Path: {release_path_str}")
    db.outputs.spawnedPrimPath = marker_path
    db.outputs.execOut = True
    return True