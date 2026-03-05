# PI Client — Isaac Sim 双臂机器人 OpenPI 客户端与数据流水线

在 Isaac Sim 中连接 OpenPI 策略服务器做视觉闭环控制，并支持采集演示数据、转换为 LeRobot 格式用于训练。

---

## 功能概览

| 功能 | 脚本 | 说明 |
|------|------|------|
| **策略执行** | `pi_client_test01.py` | 在仿真中加载双臂机器人，通过 WebSocket 连 OpenPI 服务器，用三路相机 + 14 维状态推理，将动作映射到 23 维关节并下发（含关节限位探针/夹紧） |
| **关节映射检查** | `check_joint_mapping.py` | 在 Isaac Sim 中列出机器人 DOF，按名称分类左/右臂与夹爪，输出 ALOHA/PI0 14 维映射建议 |
| **原始数据采集** | `record_raw_episode.py` | 在仿真中录制 RGB、qpos、action、prompt，保存为 `raw_episode_xxxx/` |
| **转 LeRobot 中间格式** | `convert_raw_to_lerobot.py` | 将 raw 数据转为 `steps.jsonl` + PNG，输出到 `lerobot_dataset_xxxx/episode_xxxxxx/` |
| **构建 LeRobot 数据集** | `build_lerobot_dataset.py` | 用 LeRobot API 打成标准 Parquet 数据集，输出到 `lerobot_hf_xxxx/` |

更细的数据格式与流程见 [PROJECT_INTRODUCTION.md](PROJECT_INTRODUCTION.md)。

---

## 环境要求

- **Isaac Sim**（用于 `pi_client_test01.py`、`check_joint_mapping.py`、`record_raw_episode.py`）
- **Python**：用于 `convert_raw_to_lerobot.py`、`build_lerobot_dataset.py`（需 `lerobot`、PIL、numpy 等）
- **OpenPI 服务端**：策略推理（`pi_client_test01.py` 通过 WebSocket 连接）

---

## 快速开始

### 1. 策略执行（pi_client_test01）

在 `pi_client_test01.py` 顶部修改：

- `SCENE_USD`：场景 USD 路径  
- `CAM_LEFT` / `CAM_RIGHT` / `CAM_MAIN`：三路相机 prim 路径  
- `ROBOT_ROOT`：机器人根路径  
- `HOST` / `PORT`：OpenPI 服务器地址  
- `LEFT_ARM_JOINTS` / `RIGHT_ARM_JOINTS` / `LEFT_GRIP_*` / `RIGHT_GRIP_*`：与 USD 中关节名一致  

在 Isaac Sim 的 Python 环境中运行：

```bash
python pi_client_test01.py
```

### 2. 检查关节映射

确认场景与 `ROBOT_ROOT` 后，在 Isaac Sim 中运行：

```bash
python check_joint_mapping.py
```

会打印 DOF 列表及建议的 14 维映射，便于校对 `pi_client_test01.py` 中的关节名配置。

### 3. 数据流水线（采集 → 转换 → 建库）

```bash
# 采集（在 Isaac Sim 中）
python record_raw_episode.py

# 转为 LeRobot 中间格式
python convert_raw_to_lerobot.py --raw raw_episode_0001 --out lerobot_dataset_rm75_demo --episode-index 1 --fps 30.0

# 构建最终数据集
python build_lerobot_dataset.py --episode-dir lerobot_dataset_rm75_demo/episode_000001 --out lerobot_hf_rm75_demo_v2 --dataset-name rm75_demo --fps 30.0
```

---

## 项目结构（简要）

```
pi_client/
├── README.md                    # 本文件
├── PROJECT_INTRODUCTION.md      # 数据流水线详细说明
├── pi_client_test01.py          # 双臂 OpenPI 客户端（策略执行）
├── check_joint_mapping.py       # 关节映射检查工具
├── record_raw_episode.py        # 原始数据采集
├── convert_raw_to_lerobot.py    # raw → LeRobot 中间格式
├── build_lerobot_dataset.py     # 中间格式 → LeRobot Parquet 数据集
└── ...
```

---

## 许可与依赖

- 依赖各脚本中 import 的 Isaac Sim / omni、openpi_client、lerobot 等，请按各自项目要求安装与授权。
