# ClothLauncher Script Node (World直下・起動時完全非表示対応版)
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

    db.outputs.spawnedPrimPath = marker_path
    db.outputs.execOut = True
    return True