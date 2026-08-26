# ClothLauncher Script Node (reset 自動 False リセット機能付き)
import numpy as np
from collections import deque
from pxr import Usd, UsdGeom, Gf, Vt, Sdf
from isaacsim.core.utils.prims import is_prim_path_valid
import omni.usd
import omni.graph.core as og

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

def _reset_single_marker(marker_path, stage, init_transform):
    """単一の雑巾をデフォルトの位置・物理OFF・表示・速度ゼロに初期化"""
    if not is_prim_path_valid(marker_path):
        return
    
    marker_prim = stage.GetPrimAtPath(marker_path)
    
    # 1. 物理シミュレーションを停止
    if marker_prim.HasAPI("PhysxDeformableBodyAPI"):
        attr = marker_prim.GetAttribute("physxDeformable:deformableBodyEnabled")
        if attr:
            attr.Set(False)
    
    # 2. 表示状態を固定
    imageable = UsdGeom.Imageable(marker_prim)
    if imageable:
        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    
    # 3. デフォルト（元の位置・姿勢）に戻す
    xformable = UsdGeom.Xformable(marker_prim)
    xformable.SetResetXformStack(True)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(init_transform)

    # 4. 頂点速度のリセット
    simulation_mesh_path = f"{marker_path}/simulation_mesh"
    simulation_mesh_prim = stage.GetPrimAtPath(simulation_mesh_path)
    if simulation_mesh_prim:
        simulation_mesh = UsdGeom.Mesh(simulation_mesh_prim)
        sim_points = simulation_mesh.GetPointsAttr().Get()
        if sim_points:
            num_points = len(sim_points)
            zero_vels = Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * num_points)
            simulation_mesh.GetVelocitiesAttr().Set(zero_vels)

def _reset_all_markers(state, stage):
    """すべての雑巾を記憶しておいた「元の位置・デフォルト状態」に戻す"""
    state.available = deque()

    for path, init_transform in state.initial_transforms.items():
        if not is_prim_path_valid(path):
            continue
        
        _reset_single_marker(path, stage, init_transform)
        state.available.append(path)

def setup(db):
    state = db.internal_state
    stage = omni.usd.get_context().get_stage()
    state.stage = stage
    state.initial_transforms = {}

    # inputs:reset ポートが無ければ自動作成
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

    raw_root_path = _extract_prim_path_string(db.inputs.poolRootPrim)
    if not raw_root_path:
        db.log_error("[ClothLauncher] poolRootPrim is empty or invalid")
        return True
    
    state.root_path = raw_root_path.rstrip('/')
    state.pool_size = int(db.inputs.poolSize)

    # 各雑巾の「元の位置（デフォルトのTransform）」を取得して記録する
    for i in range(state.pool_size):
        path = f"{state.root_path}/marker_{i:02d}"
        if not is_prim_path_valid(path):
            continue
        
        marker_prim = stage.GetPrimAtPath(path)
        xformable = UsdGeom.Xformable(marker_prim)
        
        local_transformation = xformable.GetLocalTransformation(Usd.TimeCode.Default())
        state.initial_transforms[path] = local_transformation

    # 全雑巾をデフォルト状態にリセット
    _reset_all_markers(state, stage)
    db.log_info(f"[ClothLauncher] Pool initialized and default transforms saved. Total: {len(state.available)}")

def compute(db):
    state = db.internal_state
    stage = getattr(state, "stage", None)
    if stage is None:
        stage = omni.usd.get_context().get_stage()
        state.stage = stage

    # --- リセット処理 (reset == True の場合) ---
    is_reset = False
    if hasattr(db.inputs, "reset"):
        is_reset = bool(db.inputs.reset)

    if is_reset:
        _reset_all_markers(state, stage)
        
        # ★ reset 入力を即座に False へ書き戻す処理
        try:
            reset_attr = db.node.get_attribute("inputs:reset")
            if reset_attr and reset_attr.is_valid():
                reset_attr.set(False)
        except Exception as e:
            db.log_warning(f"[ClothLauncher] Could not reset input attribute: {e}")

        db.log_info("[ClothLauncher] Reset executed and 'reset' attribute toggled back to False.")
        return True

    # --- 通常の射出処理 (execIn トリガー) ---
    if hasattr(db.inputs, "execIn") and not db.inputs.execIn:
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

    # プールから 1 枚取り出し
    marker_path = state.available.popleft()
    marker_prim = stage.GetPrimAtPath(marker_path)

    # 1. 射出位置のワールド座標・回転を取得
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

    # 2. ワープ移動（射出位置へ移動）
    xformable = UsdGeom.Xformable(marker_prim)
    xformable.SetResetXformStack(True)
    xformable.ClearXformOpOrder()
    rot_gf = Gf.Rotation(Gf.Quatd(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
    transform_matrix = Gf.Matrix4d().SetTransform(rot_gf, Gf.Vec3d(world_pos_np[0], world_pos_np[1], world_pos_np[2]))
    xformable.AddTransformOp().Set(transform_matrix)

    # 3. メッシュ頂点速度の設定
    simulation_mesh_path = f"{marker_path}/simulation_mesh"
    simulation_mesh_prim = stage.GetPrimAtPath(simulation_mesh_path)
    if simulation_mesh_prim:
        simulation_mesh = UsdGeom.Mesh(simulation_mesh_prim)
        sim_points = simulation_mesh.GetPointsAttr().Get()
        if sim_points:
            num_points = len(sim_points)
            velocities = Vt.Vec3fArray([Gf.Vec3f(float(world_lin_vel[0]), float(world_lin_vel[1]), float(world_lin_vel[2]))] * num_points)
            simulation_mesh.GetVelocitiesAttr().Set(velocities)

    # 4. 可視化 ＆ 物理有効化
    imageable = UsdGeom.Imageable(marker_prim)
    if imageable:
        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    if marker_prim.HasAPI("PhysxDeformableBodyAPI"):
        attr = marker_prim.GetAttribute("physxDeformable:deformableBodyEnabled")
        if attr:
            attr.Set(True)

    db.outputs.spawnedPrimPath = marker_path
    db.outputs.execOut = True
    return True