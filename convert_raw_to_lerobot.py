#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _load_prompt(raw_dir: Path) -> str:
    p = raw_dir / "prompts.jsonl"
    if not p.exists():
        return ""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            return obj.get("prompt") or obj.get("instruction") or obj.get("text") or ""
    return ""


def _sorted_rgb_files(rgb_dir: Path):
    files = list(rgb_dir.glob("*.npy"))
    if not files:
        return []
    return sorted(files, key=lambda p: int(p.stem))


def _decode_rgb_npy(npy_path: Path):
    """
    兼容几种常见存法：
    - 正常 uint8 [H,W,4]/[H,W,3]
    - object 包了一层（比如 array([img], dtype=object)）
    - 空帧：shape (0,) 或 None
    返回：np.ndarray [H,W,3] uint8 或 None（代表无效帧）
    """
    try:
        arr = np.load(npy_path, allow_pickle=True)
    except Exception:
        return None

    # 空数组
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.size == 0:
        return None

    # object 包装：array([img], dtype=object) / array(img, dtype=object)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # 可能是 shape (1,) 里面装了真正图像
        if arr.size == 1:
            arr = arr.item()
        else:
            # 乱七八糟 object，直接判无效
            return None

    if not isinstance(arr, np.ndarray):
        return None

    if arr.ndim != 3:
        return None

    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] != 3:
        return None

    # 确保 uint8
    if arr.dtype != np.uint8:
        # 有时相机会给 float [0,1]，也兼容一下
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)

    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True, help="raw_episode_xxxx 目录")
    ap.add_argument("--out", type=str, required=True, help="输出目录")
    ap.add_argument("--dataset-name", type=str, default="rm75_demo")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-skip", type=int, default=120,
                    help="最多允许跳过多少个无效 rgb 帧（防止一路空帧导致误操作）")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_npz = raw_dir / "episode.npz"
    rgb_dir = raw_dir / "rgb"
    assert episode_npz.exists(), f"找不到 {episode_npz}"
    assert rgb_dir.exists(), f"找不到 {rgb_dir}"

    prompt = _load_prompt(raw_dir)

    data = np.load(episode_npz, allow_pickle=True)

    qpos = None
    action = None
    for k in ["qpos", "state", "states", "joint_positions"]:
        if k in data.files:
            qpos = data[k]
            break
    for k in ["action", "actions", "qpos_target", "target_qpos", "joint_position_targets"]:
        if k in data.files:
            action = data[k]
            break

    if qpos is None:
        raise RuntimeError(f"episode.npz 里没找到 qpos/state 类字段，可用字段: {data.files}")
    if action is None:
        raise RuntimeError(f"episode.npz 里没找到 action 类字段，可用字段: {data.files}")

    done = data["done"] if "done" in data.files else None
    success = data["success"] if "success" in data.files else None

    rgb_files = _sorted_rgb_files(rgb_dir)
    if not rgb_files:
        raise RuntimeError(f"{rgb_dir} 里没有 .npy rgb 帧")

    T_raw = min(len(rgb_files), len(qpos), len(action))
    print(f"[INFO] rgb_files={len(rgb_files)} qpos={len(qpos)} action={len(action)} => raw T={T_raw}")

    # 输出 episode 目录
    ep_dir = out_dir / f"episode_{args.episode_index:06d}"
    images_out = ep_dir / "rgb"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    out_t = 0

    for t in range(T_raw):
        rgb = _decode_rgb_npy(rgb_files[t])
        if rgb is None:
            skipped += 1
            if skipped <= 5:
                print(f"[WARN] skip invalid rgb @ t={t}: {rgb_files[t].name}")
            if skipped > args.max_skip:
                raise RuntimeError(
                    f"跳过无效 rgb 帧太多（>{args.max_skip}）。"
                    f"请检查相机是否真的在输出，或采集脚本是否在 cam.initialize() 之后才开始写。"
                )
            continue

        png_path = images_out / f"{out_t:06d}.png"
        Image.fromarray(rgb).save(png_path)

        row = {
            "episode_index": int(args.episode_index),
            "t": int(out_t),  # 重新编号
            "raw_t": int(t),  # 保留原始帧编号，方便你 debug
            "prompt": prompt,
            "state": np.asarray(qpos[t]).astype(np.float32).tolist(),
            "action": np.asarray(action[t]).astype(np.float32).tolist(),
            "rgb_path": str(png_path.relative_to(out_dir)),
        }
        if done is not None:
            row["done"] = bool(done[t])
        if success is not None:
            row["success"] = bool(success[t]) if np.ndim(success) > 0 else bool(success)

        rows.append(row)
        out_t += 1

    ep_dir.mkdir(parents=True, exist_ok=True)
    with (ep_dir / "steps.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if out_t == 0:
        raise RuntimeError("没有任何有效 rgb 帧被写出（全部是空帧/无效帧）。请先修采集脚本。")

    meta = {
        "dataset_name": args.dataset_name,
        "episode_index": int(args.episode_index),
        "num_steps": int(out_t),
        "fps": float(args.fps),
        "state_dim": int(np.asarray(qpos[0]).shape[0]),
        "action_dim": int(np.asarray(action[0]).shape[0]),
        "skipped_invalid_rgb": int(skipped),
        "obs": {"rgb": {"type": "image", "format": "png", "path_key": "rgb_path"}},
        "state": {"type": "vector", "key": "state"},
        "action": {"type": "vector", "key": "action", "meaning": "absolute_joint_positions"},
        "prompt": {"type": "text", "key": "prompt"},
    }
    with (ep_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {ep_dir}")
    print(f"[OK] steps: {ep_dir / 'steps.jsonl'}")
    print(f"[OK] images: {images_out}")
    print(f"[OK] kept_steps={out_t}, skipped_invalid_rgb={skipped}")


if __name__ == "__main__":
    main()
