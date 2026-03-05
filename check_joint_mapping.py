#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查机器人关节与 ALOHA/PI0 模型的关节映射关系
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from pxr import UsdPhysics

# ====== 配置 ============================================
SCENE_USD = "/home/rest1/Downloads/scene.usd"
ROBOT_ROOT = "/World/test_02"
# ========================================================

# ALOHA/PI0 模型的关节定义（14 维）
ALOHA_JOINT_DEFINITION = [
    # 左臂（6 个关节）
    ("left_arm_joint_0", "左臂关节 0"),
    ("left_arm_joint_1", "左臂关节 1"),
    ("left_arm_joint_2", "左臂关节 2"),
    ("left_arm_joint_3", "左臂关节 3"),
    ("left_arm_joint_4", "左臂关节 4"),
    ("left_arm_joint_5", "左臂关节 5"),
    # 左夹爪（1 个）
    ("left_gripper", "左夹爪"),
    # 右臂（6 个关节）
    ("right_arm_joint_0", "右臂关节 0"),
    ("right_arm_joint_1", "右臂关节 1"),
    ("right_arm_joint_2", "右臂关节 2"),
    ("right_arm_joint_3", "右臂关节 3"),
    ("right_arm_joint_4", "右臂关节 4"),
    ("right_arm_joint_5", "右臂关节 5"),
    # 右夹爪（1 个）
    ("right_gripper", "右夹爪"),
]

def find_robot_articulation(stage, robot_root):
    """查找机器人的 articulation root"""
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(robot_root + "/") and p != robot_root:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    return None

def identify_joint_type(name):
    """根据关节名称识别关节类型"""
    name_lower = name.lower()
    
    # 检查是否是夹爪
    if any(keyword in name_lower for keyword in ['gripper', 'finger', 'claw', 'pincer']):
        if 'left' in name_lower or name_lower.startswith('l') or 'l_' in name_lower:
            return 'left_gripper'
        elif 'right' in name_lower or name_lower.startswith('r') or 'r_' in name_lower:
            return 'right_gripper'
        else:
            return 'gripper_unknown'
    
    # 检查是否是左臂
    if 'left' in name_lower or name_lower.startswith('l') or 'l_' in name_lower:
        return 'left_arm'
    
    # 检查是否是右臂
    if 'right' in name_lower or name_lower.startswith('r') or 'r_' in name_lower:
        return 'right_arm'
    
    return 'unknown'

def main():
    # 打开场景
    open_stage(SCENE_USD)
    
    world = World(stage_units_in_meters=1.0)
    world.reset()
    
    # 查找机器人
    stage = omni.usd.get_context().get_stage()
    robot_art_path = find_robot_articulation(stage, ROBOT_ROOT)
    
    if robot_art_path is None:
        print(f"[ERROR] 未找到机器人 articulation in {ROBOT_ROOT}")
        return
    
    robot = Articulation(prim_path=robot_art_path, name="robot")
    world.scene.add(robot)
    
    print("=" * 80)
    print("ALOHA/PI0 模型关节定义（14 维）")
    print("=" * 80)
    for i, (key, desc) in enumerate(ALOHA_JOINT_DEFINITION):
        print(f"  [{i:2d}] {key:20s} - {desc}")
    
    print("\n" + "=" * 80)
    print(f"你的机器人关节信息（共 {robot.num_dof} 个关节）")
    print("=" * 80)
    
    # 分析机器人关节
    joint_categories = {
        'left_arm': [],
        'left_gripper': [],
        'right_arm': [],
        'right_gripper': [],
        'unknown': []
    }
    
    for i, name in enumerate(robot.dof_names):
        joint_type = identify_joint_type(name)
        joint_categories[joint_type].append((i, name))
    
    # 打印分类结果
    for category, joints in joint_categories.items():
        if joints:
            print(f"\n[{category.upper()}] ({len(joints)} 个关节):")
            for idx, name in joints:
                print(f"  [{idx:2d}] {name}")
    
    # 尝试创建映射
    print("\n" + "=" * 80)
    print("关节映射建议")
    print("=" * 80)
    
    left_arm = sorted([idx for idx, _ in joint_categories['left_arm']])
    left_gripper = sorted([idx for idx, _ in joint_categories['left_gripper']])
    right_arm = sorted([idx for idx, _ in joint_categories['right_arm']])
    right_gripper = sorted([idx for idx, _ in joint_categories['right_gripper']])
    
    if len(left_arm) >= 6 and len(right_arm) >= 6:
        mapping = (
            left_arm[:6] +  # 左臂前 6 个
            (left_gripper[:1] if left_gripper else [left_arm[-1] if left_arm else 0]) +  # 左夹爪
            right_arm[:6] +  # 右臂前 6 个
            (right_gripper[:1] if right_gripper else [right_arm[-1] if right_arm else 0])  # 右夹爪
        )
        
        print("\n建议的映射关系（ALOHA 索引 -> 机器人关节索引）:")
        for aloha_idx, (key, desc) in enumerate(ALOHA_JOINT_DEFINITION):
            robot_idx = mapping[aloha_idx] if aloha_idx < len(mapping) else None
            robot_name = robot.dof_names[robot_idx] if robot_idx is not None else "N/A"
            print(f"  ALOHA[{aloha_idx:2d}] {key:20s} -> 机器人[{robot_idx:2d}] {robot_name}")
        
        print("\nPython 代码映射（可以直接复制到你的脚本中）:")
        print(f"JOINT_MAPPING = {mapping}")
    else:
        print("\n[WARN] 无法自动创建映射，因为:")
        if len(left_arm) < 6:
            print(f"  - 左臂关节不足 6 个（只有 {len(left_arm)} 个）")
        if len(right_arm) < 6:
            print(f"  - 右臂关节不足 6 个（只有 {len(right_arm)} 个）")
        print("\n请手动检查关节名称，并创建映射。")
    
    print("\n" + "=" * 80)
    print("提示：")
    print("1. 确保左臂有至少 6 个关节，右臂有至少 6 个关节")
    print("2. 确保有左夹爪和右夹爪关节（或可以映射到某个关节）")
    print("3. 关节顺序应该符合 ALOHA 的定义：左臂 -> 左夹爪 -> 右臂 -> 右夹爪")
    print("=" * 80)
    
    # 保持窗口打开
    print("\n按 Ctrl+C 退出...")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()

