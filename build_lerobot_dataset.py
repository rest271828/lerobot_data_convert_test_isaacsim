#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def resolve_image_path(ep_dir: Path, rgb_path_str: str) -> Path:
    """
    steps.jsonl 的 rgb_path 可能是：
      - "episode_000001/rgb/000000.png"   （相对 ep_dir.parent）
      - "rgb/000000.png"                 （相对 ep_dir）
      - "000000.png"                     （默认 ep_dir/rgb/）
    """
    p = Path(rgb_path_str)

    cand1 = ep_dir.parent / p
    if cand1.exists():
        return cand1

    cand2 = ep_dir / p
    if cand2.exists():
        return cand2

    cand3 = ep_dir / "rgb" / p.name
    if cand3.exists():
        return cand3

    raise FileNotFoundError(f"找不到图像: {rgb_path_str} (tried {cand1}, {cand2}, {cand3})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    ep_dir = Path(args.episode_dir)
    out = Path(args.out)
    steps_path = ep_dir / "steps.jsonl"
    assert steps_path.exists(), f"missing: {steps_path}"

    # LeRobotDataset.create 要求 out 不存在（它内部 mkdir(exist_ok=False)）
    if out.exists():
        raise RuntimeError(f"[FATAL] out 已存在，请先删除：rm -rf {out}")

    # 读第一条确定维度
    with open(steps_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    state_dim = len(first["state"])
    action_dim = len(first["action"])

    ds = LeRobotDataset.create(
        repo_id=args.dataset_name,
        root=out,
        fps=float(args.fps),
        features={
            "observation.images.rgb": {"dtype": "uint8", "shape": (64, 64, 3)},
            "observation.state": {"dtype": "float32", "shape": (state_dim,)},
            "action": {"dtype": "float32", "shape": (action_dim,)},
            "done": {"dtype": "bool", "shape": (1,)},
            "success": {"dtype": "bool", "shape": (1,)},
            # 注意：lerobot 0.4.2 会自己带上 index/frame_index/timestamp/task/task_index 的 feature
        },
    )

    # ---- 读 steps，组数据 ----
    rgbs, states, actions = [], [], []
    dones, succs = [], []
    prompts = []

    with open(steps_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            img_path = resolve_image_path(ep_dir, r["rgb_path"])
            rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            # 强制 64x64
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise RuntimeError(f"RGB shape 异常: {rgb.shape} @ {img_path}")
            if rgb.shape[0] != 64 or rgb.shape[1] != 64:
                rgb = np.array(Image.fromarray(rgb).resize((64, 64)), dtype=np.uint8)

            rgbs.append(rgb)
            states.append(np.asarray(r["state"], dtype=np.float32))
            actions.append(np.asarray(r["action"], dtype=np.float32))
            dones.append([bool(r.get("done", False))])       # (1,)
            succs.append([bool(r.get("success", False))])    # (1,)
            prompts.append(r.get("prompt", ""))

    T = len(rgbs)
    if T == 0:
        raise RuntimeError("steps.jsonl 没读到任何帧")

    # ---- lerobot 0.4.2 需要的默认字段：每帧一条（长度 T）----
    write_ep_idx = int(ds.meta.total_episodes)        # 必须从 0 递增
    start_global = int(getattr(ds.meta, "total_frames", 0))

    index = np.arange(start_global, start_global + T, dtype=np.int64)
    frame_index = np.arange(T, dtype=np.int64)
    timestamp = frame_index.astype(np.float32) / float(args.fps)

    # ✅ 关键修复：task / task_index 都做成“每帧一条”
    task_str = prompts[0] if prompts else ""
    task = [task_str] * T                      # 长度 T
    task_index = [0] * T                       # 长度 T

    episode_buffer = {
        "episode_index": int(write_ep_idx),
        "size": int(T),

        "index": index,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "task": task,
        "task_index": task_index,

        "observation.images.rgb": np.stack(rgbs, axis=0),
        "observation.state": np.stack(states, axis=0),
        "action": np.stack(actions, axis=0),
        "done": np.asarray(dones, dtype=bool),
        "success": np.asarray(succs, dtype=bool),
    }

    ds.save_episode(episode_buffer)
    #ds.save()

    print(f"[OK] wrote: {out}")
    print(f"[OK] episode_index={write_ep_idx}, frames={T}, task='{task_str}'")
    print(f"[NEXT] lerobot-dataset-viz --repo-id {out} --episodes {write_ep_idx}")


if __name__ == "__main__":
    main()
