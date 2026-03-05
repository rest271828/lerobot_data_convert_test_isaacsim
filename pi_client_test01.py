import numpy as np
import time

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation
from pxr import UsdPhysics

from omni.isaac.core.utils.types import ArticulationAction  # ✅ 关键

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi_client import image_tools


# ====== 场景 USD ======
SCENE_USD = "/home/rest1/Downloads/scene.usd"

# ====== 相机 prim path ======
CAM_LEFT  = "/World/test_02/BRXURDF0401/ArmL08_Link/ArmL08_camera"
CAM_RIGHT = "/World/test_02/BRXURDF0401/ArmR08_Link/ArmR08_camera"
CAM_MAIN  = "/World/test_02/Main_camera"

# ====== 机器人配置 ======
ROBOT_ROOT = "/World/test_02"

# ====== Pi policy server 地址 ======
HOST = "180.85.206.202"
PORT = 8000

# ====== Prompt ======
PROMPT = "Pick up the blue object and place it on the yellow table."
INFER_HZ = 30.0

# ====== DEBUG ======
DEBUG_ACTION = True
DEBUG_EVERY_N = 10

# ====== LIMIT PROBE / CLAMP ======
LIMIT_PROBE_ENABLE = True       # 打印限位探针
LIMIT_PROBE_TOPK = 3            # 打印最危险的K个
LIMIT_CLAMP_ENABLE = True  # ✅ 建议先 False 定位，确认顶限位后再开 True
LIMIT_CLAMP_SAFE_RATIO = 0.06  # 留 3% 的关节范围作为安全边际


# ------------------ 你机器人关节名字（按你输出确认是对的） ------------------
LEFT_ARM_JOINTS  = ["ArmL02_Joint", "ArmL03_Joint", "ArmL04_Joint", "ArmL05_Joint", "ArmL06_Joint", "ArmL07_Joint"]
RIGHT_ARM_JOINTS = ["ArmR02_Joint", "ArmR03_Joint", "ArmR04_Joint", "ArmR05_Joint", "ArmR06_Joint", "ArmR07_Joint"]

LEFT_GRIP_MAIN   = "JawBlock01_Joint"
LEFT_GRIP_SLAVE  = "JawBlock02_Joint"
RIGHT_GRIP_MAIN  = "JawBlock03_Joint"
RIGHT_GRIP_SLAVE = "JawBlock04_Joint"
# ------------------------------------------------------------------------------


def list_cameras():
    stage = omni.usd.get_context().get_stage()
    paths = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            paths.append(prim.GetPath().pathString)
    print("[Cameras]")
    for p in paths:
        print(" ", p)


def grab_rgb_224(cam: Camera):
    rgba = cam.get_rgba()
    if rgba is None:
        return None
    rgb = rgba[..., :3].astype(np.uint8)
    rgb = image_tools.resize_with_pad(rgb, 224, 224)
    rgb = image_tools.convert_to_uint8(rgb)
    return rgb


def find_robot_articulation(stage, robot_root):
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(robot_root + "/") and p != robot_root:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    return None


def hwc_to_chw(img):
    if img is None:
        return None
    if img.ndim == 3:
        return np.transpose(img, (2, 0, 1))
    return img


def get_limits_compat(robot):
    if hasattr(robot, "get_dof_limits"):
        try:
            lim = np.asarray(robot.get_dof_limits())
            if lim.ndim == 3:
                lim = lim[0]
            return lim[:, 0].astype(np.float32), lim[:, 1].astype(np.float32)
        except Exception:
            pass
    if hasattr(robot, "_articulation_view") and hasattr(robot._articulation_view, "get_dof_limits"):
        try:
            lim = np.asarray(robot._articulation_view.get_dof_limits())
            if lim.ndim == 3:
                lim = lim[0]
            return lim[:, 0].astype(np.float32), lim[:, 1].astype(np.float32)
        except Exception:
            pass
    return None, None


def name_to_index(dof_names, joint_name):
    try:
        return dof_names.index(joint_name)
    except ValueError:
        return -1


def build_mapping_by_name(robot):
    dof_names = list(robot.dof_names)

    left_idx  = [name_to_index(dof_names, n) for n in LEFT_ARM_JOINTS]
    right_idx = [name_to_index(dof_names, n) for n in RIGHT_ARM_JOINTS]

    lg_main = name_to_index(dof_names, LEFT_GRIP_MAIN)
    lg_slv  = name_to_index(dof_names, LEFT_GRIP_SLAVE)
    rg_main = name_to_index(dof_names, RIGHT_GRIP_MAIN)
    rg_slv  = name_to_index(dof_names, RIGHT_GRIP_SLAVE)

    print("\n[MAPPING CHECK]")
    print("left arm idx :", left_idx)
    print("right arm idx:", right_idx)
    print("left grip  main/slave :", lg_main, lg_slv)
    print("right grip main/slave :", rg_main, rg_slv)
    print("========================================\n")

    if any(i < 0 for i in left_idx) or any(i < 0 for i in right_idx):
        raise RuntimeError("找不到某些 ArmL/ArmR 关节名，请检查 dof_names 输出与你配置是否一致。")
    if lg_main < 0 or rg_main < 0 or lg_slv < 0 or rg_slv < 0:
        raise RuntimeError("找不到夹爪关节名 JawBlock01~04，请检查 dof_names。")

    # 14维顺序：L(6) + Lgrip_main(1) + R(6) + Rgrip_main(1)
    joint_mapping = left_idx + [lg_main] + right_idx + [rg_main]
    return joint_mapping, lg_main, lg_slv, rg_main, rg_slv


def grip_to_angle(val, max_open=0.041):
    # 你的日志里 grip 大概 0.24~0.29，很像 [0,1] 的开合比例
    v = float(val)
    v01 = max(0.0, min(1.0, v))
    return v01 * float(max_open)


def send_position_command(robot, controller, q_target):
    """
    ✅ 兼容不同 Isaac 版本：
    - 优先：controller.apply_action(ArticulationAction(joint_positions=...))
    - 否则：robot.set_joint_positions（兜底）
    """
    q_target = np.asarray(q_target, dtype=np.float32)

    if controller is not None and hasattr(controller, "apply_action"):
        try:
            controller.apply_action(ArticulationAction(joint_positions=q_target))
            return
        except Exception as e:
            print(f"[WARN] controller.apply_action failed: {e}")

    # 兜底
    robot.set_joint_positions(q_target)


def limit_probe_and_optional_clamp(robot, joint_mapping, qpos_now, target14, lower, upper,
                                  do_print=True, topk=3, enable_clamp=False, safe_ratio=0.03):
    """
    返回：(target14_after, did_clamp)
    - 打印14维里最危险的关节margin（q与target）
    - 可选：对 target14 做安全夹紧（留 safe_ratio 的范围边际）
    """
    idx14 = joint_mapping
    q14 = qpos_now[idx14].copy()
    t14 = target14.copy()

    low14 = lower[idx14]
    up14 = upper[idx14]
    names14 = [robot.dof_names[i] for i in idx14]

    # margin < 0 -> 越界；越接近0越贴边
    margin_q = np.minimum(q14 - low14, up14 - q14)
    margin_t = np.minimum(t14 - low14, up14 - t14)

    if do_print:
        worst_q = np.argsort(margin_q)[:topk]
        worst_t = np.argsort(margin_t)[:topk]

        print("\n[LIMIT PROBE] worst q margins:")
        for i in worst_q:
            print(f"  {i:2d} {names14[i]:20s} q={q14[i]: .6f} "
                  f"[{low14[i]: .6f}, {up14[i]: .6f}] margin={margin_q[i]: .3e}")

        print("[LIMIT PROBE] worst target margins:")
        for i in worst_t:
            print(f"  {i:2d} {names14[i]:20s} t={t14[i]: .6f} "
                  f"[{low14[i]: .6f}, {up14[i]: .6f}] margin={margin_t[i]: .3e}")

        if np.any(margin_t < 0.0):
            print("[LIMIT PROBE][BAD] target OUTSIDE limits -> 必然顶住/抖动")
        elif np.any(margin_t < 5e-3):
            print("[LIMIT PROBE][WARN] target very close to limits -> 很可能顶住/抖动")
        if np.any(margin_q < 1e-3):
            print("[LIMIT PROBE][WARN] current q at/near limits")

    did_clamp = False
    if enable_clamp:
        rng14 = (up14 - low14)
        safe = safe_ratio * rng14
        t14_clamped = np.clip(t14, low14 + safe, up14 - safe)
        did_clamp = bool(np.any(np.abs(t14_clamped - t14) > 1e-9))
        if do_print and did_clamp:
            changed = np.where(np.abs(t14_clamped - t14) > 1e-9)[0]
            print("[LIMIT CLAMP] target14 clamped (show up to 6):")
            for i in changed[:6]:
                print(f"  {i:2d} {names14[i]:20s} {t14[i]: .6f} -> {t14_clamped[i]: .6f}")
            if len(changed) > 6:
                print(f"  ... {len(changed)-6} more")
        t14 = t14_clamped

    return t14, did_clamp


def main():
    open_stage(SCENE_USD)

    world = World(stage_units_in_meters=1.0)

    stage = omni.usd.get_context().get_stage()
    list_cameras()

    robot_art_path = find_robot_articulation(stage, ROBOT_ROOT)
    if not robot_art_path:
        print("[FATAL] 未找到机器人 articulation（ArticulationRootAPI）。")
        simulation_app.close()
        return

    # ✅ 先创建/加入 scene，再 reset
    robot = Articulation(prim_path=robot_art_path, name="robot")
    world.scene.add(robot)

    # ✅ reset 让 articulation_view populate
    world.reset()
    for _ in range(3):
        world.step(render=True)

    print(f"[OK] 找到机器人 articulation: {robot_art_path}")
    print(f"[OK] 机器人 DOF 数量: {robot.num_dof}")
    print("[关节信息] robot.dof_names：")
    for i, name in enumerate(robot.dof_names):
        print(f"  [{i:2d}] {name}")

    joint_mapping, lg_main, lg_slv, rg_main, rg_slv = build_mapping_by_name(robot)

    lower, upper = get_limits_compat(robot)
    if lower is None:
        print("[WARN] 获取关节 limits 失败，LIMIT PROBE/CLAMP 将不可用。")
    else:
        print("[GRIPPER LIMITS]")
        for idx in [lg_main, lg_slv, rg_main, rg_slv]:
            print(f"  idx {idx:2d}  {robot.dof_names[idx]:20s}  lower={lower[idx]: .6f}  upper={upper[idx]: .6f}")
        print("========================================\n")

    controller = None
    try:
        controller = robot.get_articulation_controller()
        print("[OK] articulation controller 可用（将使用 apply_action）")
    except Exception as e:
        print(f"[WARN] 无法获取 articulation controller，将回退 set_joint_positions: {e}")

    # 相机
    camL = Camera(prim_path=CAM_LEFT,  name="cam_left")
    camR = Camera(prim_path=CAM_RIGHT, name="cam_right")
    camM = Camera(prim_path=CAM_MAIN,  name="cam_main")
    for c in (camL, camR, camM):
        c.initialize()

    # openpi
    print(f"[INFO] 准备连接到 openpi 服务器: {HOST}:{PORT}")
    policy = WebsocketClientPolicy(host=HOST, port=PORT)
    print("[OK] 成功连接到 openpi 服务器")
    print(f"[任务] Prompt: '{PROMPT}'")
    print(f"[配置] 推理频率: {INFER_HZ} Hz")

    # 再预热几步
    for _ in range(20):
        world.step(render=True)

    period = 1.0 / INFER_HZ
    t_next = time.time()

    while simulation_app.is_running():
        world.step(render=True)

        now = time.time()
        if now < t_next:
            continue
        t_next = now + period

        main_rgb  = grab_rgb_224(camM)
        left_rgb  = grab_rgb_224(camL)
        right_rgb = grab_rgb_224(camR)
        if main_rgb is None or left_rgb is None or right_rgb is None:
            print("camera frame None, skipping")
            continue

        # 当前关节状态
        qpos_now = robot.get_joint_positions()
        if qpos_now is None:
            continue
        qpos_now = np.asarray(qpos_now, dtype=np.float32)

        # state：按 joint_mapping 顺序取当前 qpos（14维）
        state = np.zeros((14,), dtype=np.float32)
        for i14, ridx in enumerate(joint_mapping):
            state[i14] = qpos_now[ridx]

        images = {
            "cam_high": hwc_to_chw(main_rgb),
            "cam_low": hwc_to_chw(main_rgb),
            "cam_left_wrist": hwc_to_chw(left_rgb),
            "cam_right_wrist": hwc_to_chw(right_rgb),
        }

        obs = {"images": images, "state": state, "prompt": PROMPT}
        out = policy.infer(obs)
        if "actions" not in out:
            continue

        a0 = np.asarray(out["actions"], dtype=np.float32)[0]  # (14,)

        # ===== 生成 target14（14维）=====
        mapped_now = np.array([qpos_now[i] for i in joint_mapping], dtype=np.float32)

        target14 = a0.copy()
        # 夹爪比例映射到 [0, 0.041]
        target14[6]  = grip_to_angle(a0[6],  max_open=0.041)
        target14[13] = grip_to_angle(a0[13], max_open=0.041)
        target14[13] = mapped_now[13]   # 右夹爪主关节：保持当前，不让策略控制


        # ===== DEBUG 打印 =====
        if DEBUG_ACTION:
            step_id = int(time.time() * INFER_HZ)
            if step_id % DEBUG_EVERY_N == 0:
                print("\n========== [PI0 ACTION DEBUG] ==========")
                print("a0 min/max:", float(a0.min()), float(a0.max()))
                print("mapped qpos now:", mapped_now)
                print("target14      :", target14)
                print("delta max abs :", float(np.max(np.abs(target14 - mapped_now))))
                print("========================================\n")

        # ===== LIMIT PROBE / CLAMP（插在这里：target14 已经准备好）=====
        do_probe_print = False
        if DEBUG_ACTION:
            step_id = int(time.time() * INFER_HZ)
            do_probe_print = (step_id % DEBUG_EVERY_N == 0)
        else:
            do_probe_print = True

        if LIMIT_PROBE_ENABLE and (lower is not None):
            target14, _ = limit_probe_and_optional_clamp(
                robot=robot,
                joint_mapping=joint_mapping,
                qpos_now=qpos_now,
                target14=target14,
                lower=lower,
                upper=upper,
                do_print=do_probe_print,
                topk=LIMIT_PROBE_TOPK,
                enable_clamp=LIMIT_CLAMP_ENABLE,
                safe_ratio=LIMIT_CLAMP_SAFE_RATIO,
            )

        # ===== 构造 q_target（23维）=====
        q_target = qpos_now.copy()

        # 写入 14 维目标：臂是绝对角；夹爪 target14 已经是角度
        for i14, ridx in enumerate(joint_mapping):
            q_target[ridx] = float(target14[i14])

        # slave 夹爪：负方向同步（按你的 limits 输出）
        q_target[lg_slv] = -q_target[lg_main]
        q_target[rg_slv] = -q_target[rg_main]

        # 发送命令
        send_position_command(robot, controller, q_target)

    simulation_app.close()


if __name__ == "__main__":
    main()
