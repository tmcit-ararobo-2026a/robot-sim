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

        db.log_error(
            "[ClothLauncher] poolRootPrim is empty"
        )

        return True

    pool_root_path = str(pool_root[0])

    pool_size = int(db.inputs.poolSize)

    state.available = deque()
    state.in_flight = {}

    for i in range(pool_size):

        path = f"{pool_root_path}/marker_{i:02d}"

        if not is_prim_path_valid(path):

            db.log_warning(
                f"[ClothLauncher] marker prim not found: {path}"
            )

            continue

        simulation_mesh_path = (
            f"{path}/simulation_mesh"
        )

        if not is_prim_path_valid(
            simulation_mesh_path
        ):

            db.log_warning(
                "[ClothLauncher] simulation mesh "
                f"not found: {simulation_mesh_path}"
            )

            continue

        state.available.append(path)

    db.log_info(
        "[ClothLauncher] pool ready: "
        f"{len(state.available)} markers"
    )

    return True


def compute(db):

    state = db.internal_state

    stage = state.stage

    if not state.available:

        db.log_warning(
            "[ClothLauncher] pool exhausted"
        )

        return True

    marker_path = state.available.popleft()

    # ============================================================
    # Release Point
    # ============================================================

    release_prim_target = db.inputs.releasePointPrim

    if not release_prim_target:

        db.log_error(
            "[ClothLauncher] releasePointPrim is empty"
        )

        state.available.appendleft(marker_path)

        return True

    release_path = str(
        release_prim_target[0]
    )

    release_prim = stage.GetPrimAtPath(
        release_path
    )

    if not release_prim:

        db.log_error(
            "[ClothLauncher] releasePointPrim not found: "
            f"{release_path}"
        )

        state.available.appendleft(marker_path)

        return True

    # ============================================================
    # Release Point World Transform
    # ============================================================

    xform_cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    world_mat = (
        xform_cache.GetLocalToWorldTransform(
            release_prim
        )
    )

    world_mat.Orthonormalize()

    world_pos = (
        world_mat.ExtractTranslation()
    )

    world_quat = (
        world_mat
        .ExtractRotationQuat()
        .GetNormalized()
    )

    world_pos_np = np.array(
        [
            world_pos[0],
            world_pos[1],
            world_pos[2],
        ],
        dtype=np.float64,
    )

    quat_wxyz = np.array(
        [
            world_quat.GetReal(),
            world_quat.GetImaginary()[0],
            world_quat.GetImaginary()[1],
            world_quat.GetImaginary()[2],
        ],
        dtype=np.float64,
    )

    if not (
        _is_valid_vec(world_pos_np)
        and _is_valid_quat(quat_wxyz)
    ):

        db.log_warning(
            "[ClothLauncher] invalid release pose"
        )

        state.available.appendleft(marker_path)

        return True

    quat_wxyz /= np.linalg.norm(quat_wxyz)

    # ============================================================
    # ROS2 velocity
    # ============================================================

    local_lin_vel = np.array(
        [
            db.inputs.linearVelocity[0],
            db.inputs.linearVelocity[1],
            db.inputs.linearVelocity[2],
        ],
        dtype=np.float64,
    )

    local_ang_vel = np.array(
        [
            db.inputs.angularVelocity[0],
            db.inputs.angularVelocity[1],
            db.inputs.angularVelocity[2],
        ],
        dtype=np.float64,
    )

    # ROS2 TwistがReleasePointローカル座標系である場合
    world_lin_vel = _quat_rotate(
        quat_wxyz,
        local_lin_vel,
    )

    world_ang_vel = _quat_rotate(
        quat_wxyz,
        local_ang_vel,
    )

    if not (
        _is_valid_vec(world_lin_vel)
        and _is_valid_vec(world_ang_vel)
    ):

        db.log_warning(
            "[ClothLauncher] invalid velocity"
        )

        state.available.appendleft(marker_path)

        return True

    # ============================================================
    # Move the deformable body
    # ============================================================

    marker_prim = stage.GetPrimAtPath(
        marker_path
    )

    xformable = UsdGeom.Xformable(
        marker_prim
    )

    xformable.ClearXformOpOrder()

    rot_gf = Gf.Rotation(
        Gf.Quatd(
            quat_wxyz[0],
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
        )
    )

    transform_matrix = (
        Gf.Matrix4d()
        .SetTransform(
            rot_gf,
            Gf.Vec3d(
                *world_pos_np
            ),
        )
    )

    xformable.AddTransformOp().Set(
        transform_matrix
    )

    # ============================================================
    # IMPORTANT:
    # Use simulation_mesh, NOT mesh
    # ============================================================

    simulation_mesh_path = (
        f"{marker_path}/simulation_mesh"
    )

    simulation_mesh_prim = (
        stage.GetPrimAtPath(
            simulation_mesh_path
        )
    )

    if not simulation_mesh_prim:

        db.log_error(
            "[ClothLauncher] simulation mesh not found: "
            f"{simulation_mesh_path}"
        )

        state.available.appendleft(
            marker_path
        )

        return True

    simulation_mesh = UsdGeom.Mesh(
        simulation_mesh_prim
    )

    # ============================================================
    # Simulation node positions
    # ============================================================

    sim_points_attr = (
        simulation_mesh.GetPointsAttr()
    )

    sim_points = sim_points_attr.Get()

    if not sim_points:

        db.log_error(
            "[ClothLauncher] simulation mesh "
            "has no points"
        )

        state.available.appendleft(
            marker_path
        )

        return True

    num_points = len(sim_points)

    # ============================================================
    # Simulation node velocities
    # ============================================================

    velocities = np.empty(
        (num_points, 3),
        dtype=np.float32,
    )

    # Default translational velocity
    velocities[:] = world_lin_vel

    # ============================================================
    # Angular velocity
    #
    # NOTE:
    # sim_points are in the simulation mesh local space.
    # Therefore calculate r relative to the local center.
    # ============================================================

    if np.linalg.norm(world_ang_vel) > 1e-6:

        points_np = np.asarray(
            [
                [p[0], p[1], p[2]]
                for p in sim_points
            ],
            dtype=np.float64,
        )

        center = np.mean(
            points_np,
            axis=0,
        )

        r = points_np - center

        angular_component = np.cross(
            world_ang_vel,
            r,
        )

        velocities += (
            angular_component
            .astype(np.float32)
        )

    # ============================================================
    # THIS is the important write
    # ============================================================

    sim_velocities_attr = (
        simulation_mesh.GetVelocitiesAttr()
    )

    if not sim_velocities_attr.IsValid():

        db.log_error(
            "[ClothLauncher] simulation_mesh "
            "has no velocities attribute"
        )

        state.available.appendleft(
            marker_path
        )

        return True

    sim_velocities_attr.Set(
        [
            Gf.Vec3f(
                float(v[0]),
                float(v[1]),
                float(v[2]),
            )
            for v in velocities
        ]
    )

    # ============================================================
    # Visibility
    # ============================================================

    UsdGeom.Imageable(
        marker_prim
    ).MakeVisible()

    # ============================================================
    # Debug
    # ============================================================

    db.log_info(
        "[ClothLauncher] spawned "
        f"{marker_path}"
    )

    db.log_info(
        "[ClothLauncher] simulation nodes="
        f"{num_points}"
    )

    db.log_info(
        "[ClothLauncher] velocity="
        f"{world_lin_vel}"
    )

    # ============================================================

    state.frame_counter += 1

    state.in_flight[marker_path] = {
        "released_frame":
            state.frame_counter
    }

    db.outputs.spawnedPrimPath = (
        marker_path
    )

    db.outputs.execOut = True

    return True