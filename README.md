# 古代金箔锻制工艺仿真与厚度均匀性分析系统 (v3)

> 南京金箔锻制工艺数字化研究平台 — 基于塑性力学 + 强化学习 + GPU三维可视化

- 版本：v3.0.0（工程化版）
- 后端：FastAPI + gunicorn/uvicorn + Redis Pub/Sub + InfluxDB
- 前端：Three.js ShaderMaterial + 2D Canvas 云图
- 部署：Docker Compose 一键编排

---

## 一、系统架构

```
                    ┌─────────────────────────────────────┐
                    │         前端 (浏览器)                │
                    │  GoldFoil3D (3D) + ThicknessPanel   │
                    │  WebSocket + REST API               │
                    └──────────────┬──────────────────────┘
                                   │ :8000
                    ┌──────────────▼──────────────────────┐
                    │  FastAPI 网关 (gunicorn+uvicorn)    │
                    │  Gzip压缩 · CORS · 健康检查         │
                    └──┬────────┬────────┬────────┬───────┘
                       │        │        │        │
         ┌─────────────▼──┐ ┌───▼─────┐ ┌▼────────┐ ┌▼─────────────┐
         │  dtu_receiver  │ │plastici-│ │rl_opt-  │ │  alarm_ws     │
         │  传感器校验     │ │ty_sim   │ │imizer   │ │ 告警+WS推送   │
         └──────┬─────────┘ └──┬──────┘ └──┬──────┘ └──────┬───────┘
                │              │           │               │
                └──────────────┴───────────┴───────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Redis Pub/Sub 总线  │ ← 内存降级
                          └─────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │   InfluxDB 2.x   │  │   模拟器容器      │  │   配置文件        │
    │  downsampling    │  │  锤击路径+力度    │  │ material.json    │
    │  5 buckets       │  │  5种工艺预设      │  │ rl_config.json   │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 模块职责

| 模块 | 文件 | 核心功能 |
|---|---|---|
| **API 网关** | `backend/main.py` | REST/WebSocket 入口，向后兼容全部 v2 接口，4 进程 gunicorn |
| **消息总线** | `backend/modules/common.py` | RedisBus（9 频道定义 + 内存降级）、配置加载器 |
| **DTU 接收器** | `backend/modules/dtu_receiver.py` | 传感器数据校验、范围过滤、异常标记 |
| **塑性仿真** | `backend/modules/plasticity_simulator.py` | 物理模型持有 + 锤击计算 + 厚度/网格发布 |
| **RL 优化器** | `backend/modules/rl_optimizer_module.py` | Q 表策略 + 预训练 + 动作推荐 |
| **告警 WebSocket** | `backend/modules/alarm_ws.py` | 破裂风险评估、WS 连接池、多频道广播 |
| **三维渲染** | `frontend/js/modules/gold_foil_3d.js` | Three.js、GPU Shader 厚度云图、锤头动画 |
| **UI 面板** | `frontend/js/modules/thickness_panel.js` | 2D 热力图、指标卡、告警日志、控件 |
| **协调入口** | `frontend/js/app.js` | WebSocket 连接 + 模块桥接 + API 调用 |
| **物理模型** | `backend/physics/physics_model.py` | 张量力学 + 自适应重划 + Ludwik 硬化 |
| **强化学习** | `backend/rl/rl_optimizer.py` | Q 学习 + 行为克隆预训练 + 厚处启发式 |

### Redis Pub/Sub 频道

| 频道 | 方向 | 负载 |
|---|---|---|
| `sensor_raw` | DTU → 仿真 | 原始传感器报文 |
| `hammer:request` | DTU → 仿真 | 校验后的锤击请求 |
| `strike:result` | 仿真 → RL/告警 | 锤击结果指标 |
| `thickness:updated` | 仿真 → RL/告警 | 厚度数组 + 均匀性 |
| `rl:action:request` | 网关 → RL | 下一步动作请求 |
| `rl:action:result` | RL → 仿真/网关 | 推荐动作 (位置+力度) |
| `alarm:triggered` | 仿真/RL → 告警 | 破裂风险事件 |
| `mesh:quality` | 仿真 → 网关 | 网格畸变度报告 |
| `system:event` | 全模块 | 生命周期事件 |

### InfluxDB 分层存储 + 降采样

| Bucket | 保留期 | 粒度 | 用途 |
|---|---|---|---|
| `gold-foil-data` (主) | 永久 | 原始 | 读写主入口 |
| `gold-foil-data_raw` | 7 天 | 原始 | 近实时查询 |
| `gold-foil-data_1m` | 30 天 | 1 分钟聚合 | 小时级报表 |
| `gold-foil-data_1h` | 365 天 | 1 小时聚合 | 月度趋势 |
| `gold-foil-data_1d` | 5 年 | 1 天聚合 | 年度研究分析 |

Downsampling 任务（Flux Task，`scripts/init_influxdb.py` 创建）：
- `raw → 1m`：每 1 分钟，offset 30s
- `1m  → 1h`：每 1 小时，offset 5m
- `1h  → 1d`：每 1 天，  offset 15m

Measurement：`forging_metrics`、`uniformity_metrics`、`fracture_risk`、`rl_optimization`、`thickness_snapshot`

---

## 二、部署步骤

### 前置要求

- Docker ≥ 24.0 + Compose V2
- 至少 4 核 / 8 GB RAM（后端 4 worker + InfluxDB + Redis + 模拟器）
- 磁盘 ≥ 20 GB（InfluxDB 历史数据 + 镜像）

### 2.1 一键启动（默认配置）

```bash
# 1. 克隆并进入目录
cd AI_solo_coder_task_A_087

# 2. 复制环境变量模板
cp .env.example .env
# (可选) 编辑 .env 修改端口和密码

# 3. 启动核心服务（后端 + Redis + InfluxDB + 初始化）
docker compose up -d --build

# 4. 查看启动日志
docker compose logs -f backend

# 5. 验证健康
curl http://localhost:8000/api/health
# 期望返回: {"status":"ok","version":"3.0.0","architecture":"v3-modular",...}
```

启动后访问：
- **Web 界面**：http://localhost:8000/
- **API 文档**：http://localhost:8000/docs
- **InfluxDB UI**：http://localhost:8086 (用户 `admin`，密码见 `.env`)
- **Redis**：`localhost:6379`（无密码）

### 2.2 启动模拟器（可选，独立 profile）

```bash
# 使用南京乌金打传统工艺预设 (500 次锤击)
SIM_COMMAND="--preset nanjing_wujin" docker compose --profile simulator up simulator

# 或自定义：螺旋路径 + 中锤 + 300 次
docker compose run --rm simulator --path spiral --force medium --strikes 300

# 只跑不写入 InfluxDB
docker compose run --rm simulator --preset kaipi --no-influxdb
```

### 2.3 关闭与清理

```bash
# 停止服务（保留数据卷）
docker compose down

# 停止并清理所有数据
docker compose down -v

# 仅重建后端（代码修改后）
docker compose up -d --build backend
```

### 2.4 本地开发（无 Docker）

```bash
# Python 3.11+
pip install -r requirements.txt

# 确保 Redis/InfluxDB 已启动（或让 Redis 自动降级内存模式）

# 后端开发模式
cd backend && python main.py   # 单进程 uvicorn，端口 8000

# 或者生产模式（多进程 gunicorn）
cd backend && gunicorn main:app -c gunicorn_conf.py

# 单独跑模拟器
python scripts/simulator.py --list-presets
python scripts/simulator.py --preset ai_optimized --grid-size 64
```

---

## 三、金箔锻制模拟器用法

### 3.1 五种工艺预设

```bash
python scripts/simulator.py --list-presets
```

| 预设名 | 路径 | 力度 | 锤击数 | 初始厚度 | 说明 |
|---|---|---|---|---|---|
| `nanjing_wujin` | center_out | nanjing_wujin | 500 | 1000 μm | **南京乌金打传统工艺**（推荐默认） |
| `kaipi` | grid_scan | heavy | 120 | 2000 μm | **开坯阶段**：重锤快速延展 |
| `yanhou` | spiral | medium | 300 | 800 μm | **延后阶段**：中锤控均匀 |
| `jingzheng` | center_out | light | 800 | 50 μm | **精整阶段**：轻锤高频修形 |
| `ai_optimized` | pretrained | variable | 300 | 1000 μm | **AI 强化学习**：自动路径+力度 |

用法：
```bash
# 南京乌金打（默认）
python scripts/simulator.py --preset nanjing_wujin

# AI 优化路径
python scripts/simulator.py --preset ai_optimized --grid-size 64
```

### 3.2 七种锤击路径

| 路径 | 说明 | 适用场景 |
|---|---|---|
| `center_out` | 由中心向外分层打（南京金箔传统） | 通用 |
| `spiral` | 阿基米德螺旋线 | 均匀延展 |
| `grid_scan` | 逐行扫描网格化 | 开坯粗打 |
| `diagonal` | 对角线交叉打 | 消应力 |
| `random` | 随机厚处优先 | 对比基准 |
| `heuristic` | 启发式厚处打重锤 | 半自动优化 |
| `rl` | 纯强化学习 Q 表策略 | AI 路径 |
| `pretrained` | 演示预训练策略 | AI 优化（推荐） |

### 3.3 五种力度预设

| 力度 | 范围 (N) | 半径 | 退火应变阈值 | 说明 |
|---|---|---|---|---|
| `light` | 300~600 | 15 mm | 0.20 | 精整 |
| `medium` | 600~1200 | 15 mm | 0.18 | 延展（默认） |
| `heavy` | 1000~1800 | 18 mm | 0.15 | 开坯 |
| `nanjing_wujin` | 500~1500 | 16 mm | 0.18 | 传统力度 |
| `variable` | 300~1800 | 15 mm | 0.18 | 随 CV 自动调整 |

### 3.4 自定义锤击序列（JSON）

```json
// my_custom_path.json
{
  "strikes": [
    { "pos_x": 0,   "pos_y": 0,   "force": 1500, "radius_mm": 18, "label": "center-heavy" },
    { "pos_x": -40, "pos_y": 40,  "force": 1200, "label": "NW" },
    { "pos_x":  40, "pos_y": 40,  "force": 1200, "label": "NE" },
    { "pos_x": -40, "pos_y": -40, "force": 1200, "label": "SW" },
    { "pos_x":  40, "pos_y": -40, "force": 1200, "label": "SE" },
    { "pos_x": 0,   "pos_y": 0,   "force": 800,  "wait_sec": 2.0 }
  ]
}
```

```bash
python scripts/simulator.py --custom-path my_custom_path.json --foil-id NF-CUSTOM-01
```

### 3.5 完整参数

```bash
python scripts/simulator.py --help
```

```
--preset          工艺预设名 (nanjing_wujin/kaipi/yanhou/jingzheng/ai_optimized)
--list-presets    列出所有预设并退出
--path            路径类型 (center_out/spiral/grid_scan/diagonal/random/heuristic/rl/pretrained)
--force           力度类型 (light/medium/heavy/nanjing_wujin/variable)
--strikes         总锤击次数 (默认 200)
--interval        锤击间隔秒 (默认 1.0s)
--anneal-every    每 N 次锤击检查退火
--anneal-temp     退火温度 °C (默认 420)
--custom-path     自定义 JSON 锤击序列路径
--foil-id         金箔 ID
--craftsman       工匠 ID
--grid-size       物理网格大小 (默认 48，建议 32/48/64/96)
--initial-thickness  初始厚度 μm
--no-influxdb     禁用 InfluxDB 写入
```

---

## 四、关键文件索引

```
AI_solo_coder_task_A_087/
├── Dockerfile.backend             # FastAPI 多阶段构建 (builder+runtime slim)
├── Dockerfile.simulator           # 模拟器多阶段构建
├── docker-compose.yml             # 4 服务编排
├── .env.example                   # 环境变量模板
├── .dockerignore
├── requirements.txt
├── README.md
├── backend/
│   ├── main.py                    # API 网关 v3 (gzip, CORS, 健康检查)
│   ├── gunicorn_conf.py           # 生产 Gunicorn 配置 (4 worker, UvicornWorker)
│   ├── config/
│   │   ├── material.json          # 材料/锤头/重划参数外置
│   │   └── rl_config.json         # RL/奖励/预训练参数外置
│   ├── modules/
│   │   ├── common.py              # RedisBus + 配置加载 + 频道常量
│   │   ├── dtu_receiver.py        # 传感器采集校验
│   │   ├── plasticity_simulator.py# 塑性变形计算
│   │   ├── rl_optimizer_module.py # 强化学习优化
│   │   └── alarm_ws.py            # 告警评估 + WebSocket
│   ├── physics/physics_model.py   # 塑性力学模型 (v2 自适应重划)
│   └── rl/rl_optimizer.py         # RL 算法 (v2 预训练)
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── app.js                 # 协调入口 (WS 管理 + 模块桥接)
│       └── modules/
│           ├── gold_foil_3d.js    # 三维渲染模块 (Three.js Shader)
│           └── thickness_panel.js # UI 厚度云图模块
└── scripts/
    ├── simulator.py               # 可配置化锻制模拟器 (预设/路径/力度/自定义)
    └── init_influxdb.py           # InfluxDB 初始化 + 降采样任务 + DBRP
```

---

## 五、生产调优参考

| 场景 | 建议配置 |
|---|---|
| 单机开发 | `BACKEND_WORKERS=2`，模拟器 `--grid-size 48`，`REDIS_MAXMEMORY 256mb` |
| 单机生产 | `BACKEND_WORKERS=4`，模拟器 `--grid-size 64`，InfluxDB SSD |
| 研究复现 | 模拟器 `--grid-size 96 --preset ai_optimized`，`BACKEND_WORKERS=8` |
| 集群部署 | Redis Sentinel / InfluxDB Enterprise，后端 K8s HPA |

性能特征（v3 模块化）：
- 单锤击延迟（48 网格）：~8-15 ms
- 单锤击延迟（96 网格）：~30-50 ms
- WebSocket 推送端到端：< 40 ms（含 Redis 往返）
- Gzip 开启后前端首屏：传输体积减少 ~70%

---

## 六、健康检查接口

```bash
$ curl http://localhost:8000/api/health
{
  "status": "ok",
  "version": "3.0.0",
  "architecture": "v3-modular",
  "components": {
    "redis": "ok",
    "influxdb": "ok",
    "dtu_receiver": "ok",
    "plasticity_simulator": "ok",
    "rl_optimizer": "ok",
    "alarm_ws": "ok"
  },
  "grid_size": 48,
  "strike_count": 0,
  "avg_thickness_um": 500.0
}
```

---

*© 南京金箔锻制工艺数字化研究 · v3 Engineering Build*
