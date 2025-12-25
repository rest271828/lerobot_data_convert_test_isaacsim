# Check out the standalone examples for more details on different implementations
# - Linux: ~/.local/share/ov/pkg/isaac_sim-x.x.x/standalone_examples
# - Windows: %userprofile%\AppData\Local\ov\pkg\isaac_sim-x.x.x\standalone_examples
# - Container: /isaac-sim/standalone_examples

# Import and launch the Omniverse Toolkit before any other imports.
# Note: Omniverse loads various plugins at runtime which
# cannot be imported unless the Toolkit is already running.
from isaacsim import SimulationApp

# Parse any command-line arguments specific to the standalone application
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test-arg", type=str, default="test", help="Test argument.")
# Parse only known arguments, so that any (eg) Kit settings are passed through to the core Kit app
args, _ = parser.parse_known_args()

# See DEFAULT_LAUNCHER_CONFIG for available configuration
# https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
launch_config = {"headless": False}
# Launch the Toolkit
simulation_app = SimulationApp(launch_config)

# Locate any other import statement after this point
import omni

try:
    import omni.usd
    import numpy as np
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.sensor import Camera
    from pxr import UsdPhysics

    USD_PATH = "/home/rest1/Downloads/scene.usd"

    # 1) 打开你的完整场景
    omni.usd.get_context().open_stage(USD_PATH)

    # 2) World 接管仿真
    world = World(stage_units_in_meters=1.0)

    # ---------- A) 找机器人 articulation root ----------
    stage = omni.usd.get_context().get_stage()

    # 你选中的根 Xform（不一定就是 articulation root，但我们从它下面去找）
    ROBOT_ROOT = "/World/test_02"

    # 在 /World/test_02 子树下找 ArticulationRootAPI
    art_root_path = None
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(ROBOT_ROOT + "/") and p != ROBOT_ROOT:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root_path = p
            break

    if art_root_path is None:
        raise RuntimeError(f"没在 {ROBOT_ROOT} 子树下找到 ArticulationRootAPI。请确认机器人是否是 articulation。")

    print("[OK] articulation root:", art_root_path)

    # 用 Articulation 绑定
    robot = Articulation(prim_path=art_root_path, name="robot")
    world.scene.add(robot)

    # ---------- B) 绑定相机 ----------
    CAMERA_PATH = "/World/test_02/BRXURDF0401/ArmL08_Link/ArmL08_camera"
    cam = Camera(prim_path=CAMERA_PATH, name="arm_cam")
    world.scene.add(cam)

    # 3) reset 后对象才可用（关节、相机数据）
    world.reset()
    cam.initialize()

    print("[OK] dof count:", robot.num_dof)
    print("[OK] first 10 dof names:", robot.dof_names[:10])

    # 4) 主循环：读相机 + 做一个最简单的关节摆动
    t = 0.0
    frame = 0

    while simulation_app.is_running():
        world.step(render=True)
        t += world.get_physics_dt()

        # ---- 读取相机（作为 Pi 的 obs）----
        rgba = cam.get_rgba()    # uint8, shape [H,W,4]
        depth = cam.get_depth()  # float32, shape [H,W]

        if frame % 60 == 0 and rgba is not None:
            print("[OBS] rgba:", rgba.shape, rgba.dtype, "depth:", None if depth is None else (depth.shape, depth.dtype))

        # ---- 机器人控制：让第 0 个 DOF 做正弦 ----
        q = robot.get_joint_positions()
        if q is not None and len(q) > 0:
            q_target = q.copy()
            q_target[0] = 0.5 * np.sin(t)
            robot.set_joint_positions(q_target)

        frame += 1

finally:
    simulation_app.close()