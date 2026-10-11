import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
import math

# モジュールスコープの状態変数(ノードごとにモジュールが保持される)
_state = {
    "vx_filt": 0.0,
    "vy_filt": 0.0,
    "wz_filt": 0.0,
}


def setup(db: og.Database):
    _state["vx_filt"] = 0.0
    _state["vy_filt"] = 0.0
    _state["wz_filt"] = 0.0


def cleanup(db: og.Database):
    pass


def compute(db: og.Database):
    prims = db.inputs.targetPrim
    if not prims:
        db.log_warning("targetPrim is not connected")
        return True

    prim_path = str(prims[0])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return True

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        db.log_warning(f"Invalid or non-RigidBody prim: {prim_path}")
        return True

    rigid_body = UsdPhysics.RigidBodyAPI(prim)

    dt = db.inputs.dt
    if dt <= 0.0:
        return True

    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)
    rot_mat3 = Gf.Matrix3d(world_transform.ExtractRotationMatrix())
    
    # ローカルX軸（正面方向）の成分から正確なyawを算出する
    yaw = math.atan2(rot_mat3[0][1], rot_mat3[0][0])

    # 移動方向を逆にするため、両方の符号を反転
    vx_target = db.inputs.linearVelocity[0]
    vy_target = db.inputs.linearVelocity[1]
    wz_target = -db.inputs.angularVelocity[2]

    alpha = 0.15
    _state["vx_filt"] += (vx_target - _state["vx_filt"]) * alpha
    _state["vy_filt"] += (vy_target - _state["vy_filt"]) * alpha
    _state["wz_filt"] += (wz_target - _state["wz_filt"]) * alpha

    vx = _state["vx_filt"]
    vy = _state["vy_filt"]
    wz = _state["wz_filt"]

    dx_world = vx * math.cos(yaw) - vy * math.sin(yaw)
    dy_world = vx * math.sin(yaw) + vy * math.cos(yaw)

    cur_lin_vel = rigid_body.GetVelocityAttr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
    cur_ang_vel = rigid_body.GetAngularVelocityAttr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)

    new_lin_vel = Gf.Vec3f(float(dx_world), float(dy_world), cur_lin_vel[2])
    new_ang_vel = Gf.Vec3f(cur_ang_vel[0], cur_ang_vel[1], float(math.degrees(wz)))

    rigid_body.GetVelocityAttr().Set(new_lin_vel)
    rigid_body.GetAngularVelocityAttr().Set(new_ang_vel)

    return True