import numpy as np
import time

# 1) 启动 Isaac Sim（headless=False 可以看到窗口；True 则无窗口）
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

# 2) 现在才能 import omni/isaac 相关模块
import omni.usd
from pxr import UsdGeom
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.sensor import Camera

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi_client import image_tools


# ====== 改成你的场景 USD（如果你已经有 scene.usd）======
SCENE_USD = "/home/rest1/Downloads/scene.usd"
# ========================================================

# ====== 改成你相机 prim path（从 Stage 里复制）===========
CAM_LEFT  = "/World/test_02/BRXURDF0401/ArmL08_Link/ArmL08_camera"
CAM_RIGHT = "/World/test_02/BRXURDF0401/ArmR08_Link/ArmR08_camera"
CAM_BASE  = CAM_LEFT   # 没 base 相机先顶上
# ========================================================

# ====== 改成你 Pi policy server 地址 =====================
HOST = "192.168.1.50"
PORT = 8000
# ========================================================

PROMPT = "pick up the object and place it on the table"
INFER_HZ = 5.0   # 推理频率：5Hz 先稳一点


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


def main():
    # 打开你的场景
    open_stage(SCENE_USD)

    world = World(stage_units_in_meters=1.0)
    world.reset()

    # 可选：打印出当前场景的所有相机，方便你填路径
    list_cameras()

    camL = Camera(prim_path=CAM_LEFT,  name="cam_left")
    camR = Camera(prim_path=CAM_RIGHT, name="cam_right")
    camB = Camera(prim_path=CAM_BASE,  name="cam_base")
    for c in (camL, camR, camB):
        c.initialize()

    policy = WebsocketClientPolicy(host=HOST, port=PORT)

    # 预热几帧，避免黑屏
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

        base_rgb  = grab_rgb_224(camB)
        left_rgb  = grab_rgb_224(camL)
        right_rgb = grab_rgb_224(camR)
        if base_rgb is None or left_rgb is None or right_rgb is None:
            print("camera frame None, skipping")
            continue

        # 先用占位 state 跑通；后面你再替换成真实机器人状态
        state = np.zeros((10,), dtype=np.float32)

        obs = {
            "base_0_rgb": base_rgb,
            "left_wrist_0_rgb": left_rgb,
            "right_wrist_0_rgb": right_rgb,
            "state": state,
            "prompt": PROMPT,
        }

        action = policy.infer(obs)
        keys = list(action.keys())
        print("got action keys:", keys)
        if "actions" in action:
            a = np.asarray(action["actions"])
            print("actions shape:", a.shape)

    simulation_app.close()


if __name__ == "__main__":
    main()
