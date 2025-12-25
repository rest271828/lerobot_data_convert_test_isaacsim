from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # 要看窗口就 False

import os
import time
import json
import numpy as np
import omni
import omni.usd
from pxr import UsdPhysics

# 尽量用新包名（你日志里已经加载了 isaacsim.core.api）
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.sensor import Camera


USD_PATH = "/home/rest1/Downloads/scene.usd"
ROBOT_ROOT = "/World/test_02"
CAMERA_PATH = "/World/test_02/BRXURDF0401/ArmL08_Link/ArmL08_camera"

OUT_DIR = "/home/rest1/pi_client/raw_episode_0001"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "rgb"), exist_ok=True)

PROMPT = "pick up the object and place it on the target"  # 你也可以后面改成从命令行传

def find_articulation_root(stage, robot_root):
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(robot_root + "/") and p != robot_root:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    return None

try:
    # 1) 打开场景
    omni.usd.get_context().open_stage(USD_PATH)
    stage = omni.usd.get_context().get_stage()

    # 2) World
    world = World(stage_units_in_meters=1.0)

    # 3) 绑定机器人
    art_root_path = find_articulation_root(stage, ROBOT_ROOT)
    if art_root_path is None:
        raise RuntimeError(f"没找到 ArticulationRootAPI in {ROBOT_ROOT}")

    robot = Articulation(prim_path=art_root_path, name="robot")
    world.scene.add(robot)

    # 4) 绑定相机
    cam = Camera(prim_path=CAMERA_PATH, name="cam")
    world.scene.add(cam)

    # 5) reset / init
    world.reset()
    cam.initialize()

    dt = world.get_physics_dt()
    seconds = 10.0
    steps = int(seconds / dt)

    print("[OK] dt:", dt, "steps:", steps, "dof:", robot.num_dof)

    # 6) 录制缓冲
    qpos_list = []
    action_list = []   # 按你要求：action = 绝对关节位置（目标位置）
    prompt_list = []
    ts_list = []

    # 简单策略：让第 0 个关节做正弦，作为 action target
    t = 0.0

    for k in range(steps):
        world.step(render=True)
        t += dt

        # 读 state
        qpos = robot.get_joint_positions()
        if qpos is None:
            continue
        qpos = np.array(qpos, dtype=np.float32)

        # 设 action（绝对关节位置）
        q_target = qpos.copy()
        q_target[0] = 0.5 * np.sin(t)
        robot.set_joint_positions(q_target)

        # 读 rgb
        rgba = cam.get_rgba()
        if rgba is None:
            continue
        rgb = rgba[..., :3]  # uint8 HWC3

        # 写图片（frame_000001.png）
        # 用 numpy 保存最省依赖：写成 .npy 也行；这里用 .npy 最稳不需要 opencv/pillow
        np.save(os.path.join(OUT_DIR, "rgb", f"{k:06d}.npy"), rgb)

        # 记录数组
        qpos_list.append(qpos)
        action_list.append(q_target.astype(np.float32))
        prompt_list.append(PROMPT)
        ts_list.append(time.time())

        if k % 60 == 0:
            print(f"[REC] {k}/{steps} rgb={rgb.shape} qpos={qpos.shape}")

    # 7) 保存 meta（一个 episode）
    np.savez_compressed(
        os.path.join(OUT_DIR, "episode.npz"),
        qpos=np.stack(qpos_list),
        action=np.stack(action_list),
        timestamp=np.array(ts_list, dtype=np.float64),
    )
    with open(os.path.join(OUT_DIR, "prompts.jsonl"), "w") as f:
        for p in prompt_list:
            f.write(json.dumps({"prompt": p}) + "\n")

    print("[DONE] saved to", OUT_DIR)

finally:
    try:
        # 尽量先停仿真
        try:
            world.stop()
        except Exception:
            pass

        # 主动断开引用，避免 __del__ 在 shutdown 期间乱序触发
        robot = None
        cam = None
        world = None

        import gc
        gc.collect()
    finally:
        simulation_app.close()

