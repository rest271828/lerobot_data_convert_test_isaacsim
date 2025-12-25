# PI 系列机器人数据采集与 LeRobot 转换项目

## 项目概述

本项目实现了在 Isaac Sim 仿真环境中采集 PI 系列机器人数据，并将其转换为 LeRobot 训练格式的完整流程。项目包含三个核心步骤：**数据采集** → **格式转换** → **数据集构建**。

---

## 一、数据采集 (`record_raw_episode.py`)

### 1.1 核心流程

数据采集脚本在 Isaac Sim 中运行，实时录制机器人的状态、动作和视觉观测。

#### **关键步骤：**

```76:109:record_raw_episode.py
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
```

### 1.2 采集的数据类型

1. **RGB 图像**：从相机传感器获取，保存为 `.npy` 格式（`uint8 [H,W,3]`）
2. **关节位置 (qpos)**：机器人的当前关节角度，维度为 23
3. **动作 (action)**：目标关节位置（绝对位置控制），维度为 23
4. **任务提示 (prompt)**：文本描述，如 "pick up the object and place it on the target"
5. **时间戳**：每帧的采集时间

### 1.3 数据存储格式

采集的数据保存在 `raw_episode_xxxx/` 目录下：

```
raw_episode_0001/
├── episode.npz          # 压缩的 numpy 数组（qpos, action, timestamp）
├── prompts.jsonl        # 每帧的任务提示
└── rgb/
    ├── 000000.npy       # 第 0 帧 RGB 图像
    ├── 000001.npy       # 第 1 帧 RGB 图像
    └── ...
```

#### **关键保存代码：**

```111:119:record_raw_episode.py
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
```

---

## 二、格式转换 (`convert_raw_to_lerobot.py`)

### 2.1 转换目标

将原始采集数据（`.npy` 图像 + `episode.npz`）转换为 LeRobot 中间格式，生成 `steps.jsonl` 和 `meta.json`。

### 2.2 核心转换逻辑

#### **步骤 1：读取原始数据**

```107:127:convert_raw_to_lerobot.py
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
```

#### **步骤 2：解码 RGB 图像并转换为 PNG**

关键函数 `_decode_rgb_npy` 处理多种图像格式：

```34:82:convert_raw_to_lerobot.py
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
```

#### **步骤 3：生成 steps.jsonl**

每帧数据转换为 JSON 行格式：

```144:180:convert_raw_to_lerobot.py
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
```

### 2.3 输出格式

转换后的数据保存在 `lerobot_dataset_xxxx/episode_xxxxxx/` 目录：

```
lerobot_dataset_rm75_demo/
└── episode_000001/
    ├── meta.json         # 数据集元数据
    ├── steps.jsonl       # 每帧数据（JSON Lines 格式）
    └── rgb/
        ├── 000000.png    # PNG 格式图像
        ├── 000001.png
        └── ...
```

---

## 三、构建 LeRobot 数据集 (`build_lerobot_dataset.py`)

### 3.1 使用 LeRobot API

这一步使用 LeRobot 官方 API 将中间格式转换为标准数据集格式（支持 HuggingFace）。

#### **关键步骤 1：创建数据集结构**

```61:73:build_lerobot_dataset.py
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
```

#### **关键步骤 2：读取并预处理数据**

```80:98:build_lerobot_dataset.py
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
```

#### **关键步骤 3：构建 Episode Buffer 并保存**

```104:134:build_lerobot_dataset.py
    # ---- lerobot 0.4.2 需要的默认字段：每帧一条（长度 T）----
    write_ep_idx = int(ds.meta.total_episodes)        # 必须从 0 递增
    start_global = int(getattr(ds.meta, "total_frames", 0))

    index = np.arange(start_global, start_global + T, dtype=np.int64)
    frame_index = np.arange(T, dtype=np.int64)
    timestamp = frame_index.astype(np.float32) / float(args.fps)

    # ✅ 关键修复：task / task_index 都做成"每帧一条"
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
```

### 3.2 输出格式

最终数据集保存在 `lerobot_hf_xxxx/` 目录，使用 Parquet 格式：

```
lerobot_hf_rm75_demo_v2/
├── data/
│   └── chunk-000/
│       └── file-000.parquet    # 数据文件
└── meta/
    ├── info.json               # 数据集信息
    ├── stats.json              # 统计信息
    ├── tasks.parquet           # 任务信息
    └── episodes/
        └── chunk-000/
            └── file-000.parquet
```

---

## 四、数据流程总结

```
Isaac Sim 仿真
    ↓
[record_raw_episode.py]
    ↓
raw_episode_xxxx/
    ├── episode.npz (qpos, action, timestamp)
    ├── prompts.jsonl
    └── rgb/*.npy
    ↓
[convert_raw_to_lerobot.py]
    ↓
lerobot_dataset_xxxx/episode_xxxxxx/
    ├── meta.json
    ├── steps.jsonl
    └── rgb/*.png
    ↓
[build_lerobot_dataset.py]
    ↓
lerobot_hf_xxxx/ (Parquet 格式)
    └── 标准 LeRobot 数据集
```

---

## 五、关键技术点

### 5.1 数据采集
- **实时同步**：每帧同时采集 RGB、关节状态和动作
- **格式选择**：使用 `.npy` 保存图像，避免依赖图像库
- **错误处理**：跳过无效帧，记录时间戳

### 5.2 格式转换
- **兼容性**：`_decode_rgb_npy` 函数处理多种图像格式（RGBA、float、object 包装等）
- **数据验证**：自动跳过无效 RGB 帧，防止数据损坏
- **路径解析**：支持多种相对路径格式

### 5.3 数据集构建
- **图像预处理**：统一调整为 64×64 分辨率
- **字段映射**：正确映射 LeRobot 要求的字段（observation.images.rgb, observation.state, action 等）
- **元数据管理**：自动生成 index、frame_index、timestamp、task 等字段

---

## 六、使用示例

### 采集数据
```bash
python record_raw_episode.py
```

### 转换为中间格式
```bash
python convert_raw_to_lerobot.py \
    --raw raw_episode_0001 \
    --out lerobot_dataset_rm75_demo \
    --episode-index 1 \
    --fps 30.0
```

### 构建最终数据集
```bash
python build_lerobot_dataset.py \
    --episode-dir lerobot_dataset_rm75_demo/episode_000001 \
    --out lerobot_hf_rm75_demo_v2 \
    --dataset-name rm75_demo \
    --fps 30.0
```

---

## 七、项目成果

- ✅ 成功采集 600 帧数据（10 秒，30 FPS）
- ✅ 转换为 LeRobot 格式（597 帧有效数据）
- ✅ 生成标准 HuggingFace 数据集
- ✅ 支持多相机（base、left_wrist、right_wrist）
- ✅ 包含任务提示（prompt）支持

数据集可用于训练基于视觉的机器人策略模型。

