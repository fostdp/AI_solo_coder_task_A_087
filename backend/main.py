"""
金箔锻制工艺仿真系统 - FastAPI后端
功能:
  1. REST API - 锻制控制、数据查询、状态管理
  2. WebSocket - 实时告警推送、状态更新
  3. InfluxDB - 时序数据存储与查询
  4. 静态文件服务 - 前端Three.js可视化
"""
import sys
import os
import time
import json
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Query,
    Body,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics.physics_model import (
    GoldFoilPhysicsModel,
    HammerParameters,
    MaterialProperties,
)
from rl.rl_optimizer import (
    RLSession,
    ActionType,
    RLConfig,
)

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "gold-foil-simulation-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "craftsman-research")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "gold-foil-data")

FRACTURE_THRESHOLD_UM = 0.1
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


class StrikeMode(str, Enum):
    MANUAL = "manual"
    HEURISTIC = "heuristic"
    RL = "rl"


class HammerRequest(BaseModel):
    force_N: float = Field(500.0, ge=100, le=3000, description="锤击力度 (N)")
    position_x_mm: float = Field(0.0, description="锤击X坐标 (mm)")
    position_y_mm: float = Field(0.0, description="锤击Y坐标 (mm)")
    radius_mm: float = Field(15.0, description="锤头半径 (mm)")


class AnnealRequest(BaseModel):
    temperature_c: float = Field(400.0, ge=100, le=900, description="退火温度 (°C)")
    duration_min: float = Field(10.0, ge=1, le=120, description="退火持续时间 (分钟)")


class SimulationConfig(BaseModel):
    grid_size: int = Field(48, ge=16, le=128, description="物理网格大小")
    initial_thickness_um: float = Field(500.0, ge=100, le=2000, description="初始厚度 (μm)")
    rl_grid_size: int = Field(8, ge=4, le=16, description="RL策略网格大小")
    target_thickness_um: float = Field(0.5, ge=0.05, le=50, description="目标厚度 (μm)")


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.all_connections: List[WebSocket] = []
        self._lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket, channel: str = "default"):
        await websocket.accept()
        with self._lock:
            self.all_connections.append(websocket)
            if channel not in self.active_connections:
                self.active_connections[channel] = []
            self.active_connections[channel].append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.all_connections:
                self.all_connections.remove(websocket)
            for channel in self.active_connections:
                if websocket in self.active_connections[channel]:
                    self.active_connections[channel].remove(websocket)
    
    async def broadcast(self, message: dict, channel: str = None):
        if channel:
            connections = self.active_connections.get(channel, [])
        else:
            connections = list(self.all_connections)
        
        disconnected = []
        for conn in connections:
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)
        
        for conn in disconnected:
            self.disconnect(conn)


class GoldFoilService:
    """金箔锻制服务 - 单例管理物理模型和强化学习会话"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.physics = None
        self.rl_session = None
        self.config = SimulationConfig()
        self.foil_id = "NF-LIVE-001"
        self.craftsman_id = "master_wu"
        self.session_id = f"session-{int(time.time())}"
        self.strike_history: List[dict] = []
        self.alert_history: List[dict] = []
        self.auto_sim_running = False
        self.auto_sim_thread = None
        
        self._init_influxdb()
        self.reset()
    
    def _init_influxdb(self):
        """初始化InfluxDB连接"""
        try:
            self.influx_client = influxdb_client.InfluxDBClient(
                url=INFLUXDB_URL,
                token=INFLUXDB_TOKEN,
                org=INFLUXDB_ORG
            )
            self.write_api = self.influx_client.write_api(
                write_options=SYNCHRONOUS
            )
            self.query_api = self.influx_client.query_api()
            self.influxdb_available = True
        except Exception as e:
            print(f"[WARN] InfluxDB不可用: {e}")
            self.influxdb_available = False
            self.influx_client = None
            self.write_api = None
            self.query_api = None
    
    def reset(self, config: SimulationConfig = None):
        """重置仿真状态"""
        with self._lock:
            if config:
                self.config = config
            
            material = MaterialProperties(
                initial_thickness_um=self.config.initial_thickness_um,
            )
            self.physics = GoldFoilPhysicsModel(
                grid_size=self.config.grid_size,
                foil_size_mm=150.0,
                material=material,
            )
            
            rl_config = RLConfig(
                grid_size=self.config.rl_grid_size,
                force_levels=5,
                min_force=300.0,
                max_force=1200.0,
                target_thickness_um=self.config.target_thickness_um,
            )
            self.rl_session = RLSession(
                physics_model=self.physics,
                config=rl_config,
            )
            
            self.session_id = f"session-{int(time.time())}"
            self.strike_history = []
            self.alert_history = []
    
    def apply_strike(self, hammer: HammerParameters) -> dict:
        """执行锤击并持久化数据"""
        with self._lock:
            result = self.physics.apply_hammer_strike(hammer)
            thickness_data = self.physics.get_thickness_distribution()
            fracture_risk = self.physics.check_fracture_risk(FRACTURE_THRESHOLD_UM)
            
            self._persist_strike(result, thickness_data, fracture_risk)
            
            response = {
                "strike": result,
                "metrics": thickness_data["metrics"],
                "fracture_risk": fracture_risk,
                "timestamp": datetime.now().isoformat(),
            }
            
            self.strike_history.append(response)
            if len(self.strike_history) > 1000:
                self.strike_history = self.strike_history[-1000:]
            
            if fracture_risk["risk_level"] != "none":
                alert = {
                    "type": "fracture_warning",
                    "level": fracture_risk["risk_level"],
                    "message": f"厚度低于{FRACTURE_THRESHOLD_UM}μm，破裂风险：{fracture_risk['risk_level'].upper()}",
                    "risk": fracture_risk,
                    "timestamp": datetime.now().isoformat(),
                }
                self.alert_history.append(alert)
                response["alert"] = alert
            
            return response
    
    def apply_rl_step(self, mode: ActionType = ActionType.HEURISTIC) -> dict:
        """执行强化学习一步锤击"""
        with self._lock:
            action, strike_result, reward, fracture_risk = self.rl_session.step(mode=mode)
            thickness_data = self.physics.get_thickness_distribution()
            
            self._persist_strike(
                strike_result,
                thickness_data,
                fracture_risk,
                rl_action=action,
                rl_reward=reward
            )
            
            response = {
                "action": {
                    "force_N": action.force,
                    "position_mm": list(action.position),
                    "radius_mm": action.radius_mm,
                },
                "strike": strike_result,
                "metrics": thickness_data["metrics"],
                "fracture_risk": fracture_risk,
                "rl_reward": reward,
                "rl_stats": self.rl_session.policy.get_policy_stats(),
                "timestamp": datetime.now().isoformat(),
            }
            
            self.strike_history.append(response)
            
            if fracture_risk["risk_level"] != "none":
                alert = {
                    "type": "fracture_warning",
                    "level": fracture_risk["risk_level"],
                    "message": f"厚度低于{FRACTURE_THRESHOLD_UM}μm，破裂风险：{fracture_risk['risk_level'].upper()}",
                    "risk": fracture_risk,
                    "timestamp": datetime.now().isoformat(),
                }
                self.alert_history.append(alert)
                response["alert"] = alert
            
            return response
    
    def apply_annealing(self, temp_c: float, duration_min: float) -> dict:
        """执行退火"""
        with self._lock:
            result = self.physics.apply_annealing(temp_c, duration_min)
            return {
                "annealing": result,
                "metrics": self.physics.get_uniformity_metrics(),
                "timestamp": datetime.now().isoformat(),
            }
    
    def get_state(self) -> dict:
        """获取当前状态"""
        with self._lock:
            thickness_data = self.physics.get_thickness_distribution()
            fracture_risk = self.physics.check_fracture_risk(FRACTURE_THRESHOLD_UM)
            
            return {
                "foil_id": self.foil_id,
                "session_id": self.session_id,
                "craftsman": self.craftsman_id,
                "total_strikes": self.physics.strike_count,
                "total_elongation": self.physics.total_elongation,
                "thickness_distribution": thickness_data,
                "fracture_risk": fracture_risk,
                "temperature_c": float(self.physics.temperature_c.mean()),
                "plastic_strain": float(self.physics.plastic_strain.mean()),
                "auto_sim_running": self.auto_sim_running,
                "config": self.config.model_dump(),
                "recent_alerts": self.alert_history[-20:],
                "rl_stats": self.rl_session.policy.get_policy_stats() if self.rl_session else None,
            }
    
    def get_thickness_visualization(self) -> dict:
        """获取厚度可视化数据"""
        with self._lock:
            h = self.physics.thickness_um
            h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)
            
            return {
                "grid_size": self.physics.grid_size,
                "foil_size_mm": self.physics.foil_size_mm,
                "thickness_um": h.tolist(),
                "normalized": h_norm.tolist(),
                "min_um": float(h.min()),
                "max_um": float(h.max()),
                "mean_um": float(h.mean()),
                "std_um": float(h.std()),
            }
    
    def query_history(
        self,
        measurement: str = "forging_metrics",
        window_minutes: int = 60,
        limit: int = 1000,
    ) -> list:
        """从InfluxDB查询历史数据"""
        if not self.influxdb_available:
            return self.strike_history[-limit:]
        
        try:
            start_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            flux_query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                    |> range(start: {int(start_time.timestamp())}, stop: now())
                    |> filter(fn: (r) => r._measurement == "{measurement}")
                    |> filter(fn: (r) => r.foil_id == "{self.foil_id}")
                    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> sort(columns: ["_time"], desc: true)
                    |> limit(n: {limit})
            '''
            result = self.query_api.query(flux_query)
            
            records = []
            for table in result:
                for record in table.records:
                    records.append(record.values)
            return records
        except Exception as e:
            print(f"[ERROR] 查询失败: {e}")
            return self.strike_history[-limit:]
    
    def _persist_strike(
        self,
        strike_result: dict,
        thickness_data: dict,
        fracture_risk: dict,
        rl_action=None,
        rl_reward: float = 0.0,
    ):
        """持久化锤击数据到InfluxDB"""
        if not self.influxdb_available or self.write_api is None:
            return
        
        now = datetime.now(timezone.utc)
        
        try:
            metrics_point = influxdb_client.Point("forging_metrics") \
                .tag("foil_id", self.foil_id) \
                .tag("session_id", self.session_id) \
                .tag("craftsman", self.craftsman_id) \
                .field("hammer_force", strike_result["hammer_force_N"]) \
                .field("temperature", strike_result["avg_temperature_c"]) \
                .field("avg_thickness", strike_result["avg_thickness_um"]) \
                .field("min_thickness", strike_result["min_thickness_um"]) \
                .field("max_thickness", strike_result["max_thickness_um"]) \
                .field("thickness_std", strike_result["thickness_std_um"]) \
                .field("elongation_rate", strike_result["elongation_rate"]) \
                .field("total_elongation", strike_result["total_elongation"]) \
                .time(now)
            
            uniformity = thickness_data["metrics"]
            uniform_point = influxdb_client.Point("uniformity_metrics") \
                .tag("foil_id", self.foil_id) \
                .tag("session_id", self.session_id) \
                .field("coefficient_of_variation", uniformity["coefficient_of_variation"]) \
                .field("uniformity_within_5pct", uniformity["uniformity_within_5pct"]) \
                .field("uniformity_within_10pct", uniformity["uniformity_within_10pct"]) \
                .field("range_ratio", uniformity["range_ratio"]) \
                .time(now)
            
            risk_point = influxdb_client.Point("fracture_risk") \
                .tag("foil_id", self.foil_id) \
                .tag("session_id", self.session_id) \
                .tag("risk_level", fracture_risk["risk_level"]) \
                .field("risk_count", fracture_risk["risk_count"]) \
                .field("risk_fraction", fracture_risk["risk_fraction"]) \
                .field("min_thickness_um", fracture_risk["min_thickness_um"]) \
                .time(now)
            
            points = [metrics_point, uniform_point, risk_point]
            
            if rl_action is not None:
                rl_point = influxdb_client.Point("rl_optimization") \
                    .tag("foil_id", self.foil_id) \
                    .tag("session_id", self.session_id) \
                    .field("rl_reward", rl_reward) \
                    .field("rl_action_force_N", rl_action.force) \
                    .field("rl_action_x_mm", rl_action.position[0]) \
                    .field("rl_action_y_mm", rl_action.position[1]) \
                    .time(now)
                points.append(rl_point)
            
            self.write_api.write(
                bucket=INFLUXDB_BUCKET,
                org=INFLUXDB_ORG,
                record=points
            )
        except Exception as e:
            print(f"[WARN] 写入InfluxDB失败: {e}")


app = FastAPI(
    title="金箔锻制工艺仿真与厚度均匀性分析系统",
    description="基于塑性力学与强化学习的南京金箔锻制工艺研究平台",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = GoldFoilService()
ws_manager = ConnectionManager()


def get_service():
    return service


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "influxdb": "connected" if service.influxdb_available else "disconnected",
        "active_ws_connections": len(ws_manager.all_connections),
    }


@app.get("/api/state")
async def get_system_state():
    """获取完整系统状态"""
    return service.get_state()


@app.get("/api/visualization/thickness")
async def get_thickness_viz():
    """获取厚度可视化数据"""
    return service.get_thickness_visualization()


@app.get("/api/metrics/uniformity")
async def get_uniformity_metrics():
    """获取均匀性指标"""
    state = service.get_state()
    return state["thickness_distribution"]["metrics"]


@app.get("/api/risk/fracture")
async def get_fracture_risk():
    """获取破裂风险"""
    return service.get_state()["fracture_risk"]


@app.post("/api/strike")
async def apply_hammer_strike(req: HammerRequest):
    """手动执行一次锤击"""
    hammer = HammerParameters(
        force=req.force_N,
        position=(req.position_x_mm, req.position_y_mm),
        radius_mm=req.radius_mm,
    )
    result = service.apply_strike(hammer)
    
    if "alert" in result:
        await ws_manager.broadcast({
            "channel": "alerts",
            "data": result["alert"]
        }, channel="alerts")
    
    await ws_manager.broadcast({
        "channel": "state_update",
        "data": result
    })
    
    return result


@app.post("/api/strike/auto")
async def apply_auto_strike(
    mode: StrikeMode = Query(StrikeMode.HEURISTIC, description="锤击模式"),
):
    """自动锤击一步（启发式或强化学习）"""
    action_type = ActionType.HEURISTIC
    if mode == StrikeMode.RL:
        action_type = ActionType.Q_LEARNING
    
    result = service.apply_rl_step(mode=action_type)
    
    if "alert" in result:
        await ws_manager.broadcast({
            "channel": "alerts",
            "data": result["alert"]
        }, channel="alerts")
    
    await ws_manager.broadcast({
        "channel": "state_update",
        "data": result
    })
    
    return result


@app.post("/api/anneal")
async def perform_annealing(req: AnnealRequest):
    """执行退火处理"""
    result = service.apply_annealing(req.temperature_c, req.duration_min)
    await ws_manager.broadcast({
        "channel": "state_update",
        "data": {"event": "annealing", **result}
    })
    return result


@app.post("/api/reset")
async def reset_simulation(config: Optional[SimulationConfig] = None):
    """重置仿真"""
    service.reset(config)
    result = {
        "message": "仿真已重置",
        "state": service.get_state()
    }
    await ws_manager.broadcast({
        "channel": "state_update",
        "data": {"event": "reset", **result}
    })
    return result


@app.get("/api/history")
async def get_history(
    measurement: str = Query("forging_metrics", description="测量类型"),
    window_minutes: int = Query(60, ge=1, le=1440, description="时间窗口(分钟)"),
    limit: int = Query(500, ge=1, le=5000, description="返回条数"),
):
    """查询历史数据"""
    return service.query_history(measurement, window_minutes, limit)


@app.get("/api/alerts")
async def get_alerts(limit: int = Query(50, ge=1, le=500)):
    """获取告警历史"""
    return service.alert_history[-limit:]


@app.post("/api/simulation/auto/start")
async def start_auto_simulation(
    interval_sec: float = Query(1.0, ge=0.1, le=10.0, description="锤击间隔(秒)"),
    max_strikes: Optional[int] = Query(None, description="最大锤击次数"),
    mode: StrikeMode = Query(StrikeMode.HEURISTIC, description="锤击模式"),
):
    """启动自动仿真循环"""
    if service.auto_sim_running:
        raise HTTPException(status_code=400, detail="自动仿真已在运行")
    
    service.auto_sim_running = True
    action_type = ActionType.HEURISTIC if mode == StrikeMode.HEURISTIC else ActionType.Q_LEARNING
    
    def sim_loop():
        count = 0
        try:
            while service.auto_sim_running:
                if max_strikes and count >= max_strikes:
                    break
                
                result = service.apply_rl_step(mode=action_type)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if "alert" in result:
                        loop.run_until_complete(ws_manager.broadcast({
                            "channel": "alerts",
                            "data": result["alert"]
                        }, channel="alerts"))
                    
                    loop.run_until_complete(ws_manager.broadcast({
                        "channel": "state_update",
                        "data": result
                    }))
                finally:
                    loop.close()
                
                if result["fracture_risk"]["risk_level"] == "high":
                    time.sleep(2)
                
                if result["metrics"]["mean_thickness_um"] < 0.12:
                    break
                
                time.sleep(interval_sec)
                count += 1
        finally:
            service.auto_sim_running = False
    
    service.auto_sim_thread = threading.Thread(target=sim_loop, daemon=True)
    service.auto_sim_thread.start()
    
    return {
        "message": "自动仿真已启动",
        "interval_sec": interval_sec,
        "max_strikes": max_strikes,
        "mode": mode.value,
    }


@app.post("/api/simulation/auto/stop")
async def stop_auto_simulation():
    """停止自动仿真"""
    if not service.auto_sim_running:
        raise HTTPException(status_code=400, detail="自动仿真未在运行")
    
    service.auto_sim_running = False
    if service.auto_sim_thread:
        service.auto_sim_thread.join(timeout=2)
    
    return {"message": "自动仿真已停止"}


@app.get("/api/stats/summary")
async def get_stats_summary():
    """统计摘要"""
    state = service.get_state()
    metrics = state["thickness_distribution"]["metrics"]
    risk = state["fracture_risk"]
    
    return {
        "total_strikes": state["total_strikes"],
        "total_elongation": state["total_elongation"],
        "current_thickness": {
            "mean_um": metrics["mean_thickness_um"],
            "std_um": metrics["std_thickness_um"],
            "cv": metrics["coefficient_of_variation"],
        },
        "uniformity": {
            "within_5pct": metrics["uniformity_within_5pct"],
            "within_10pct": metrics["uniformity_within_10pct"],
        },
        "fracture_risk": risk,
        "temperature_c": state["temperature_c"],
        "plastic_strain": state["plastic_strain"],
        "alerts_count": len(service.alert_history),
        "in_progress": state["auto_sim_running"],
    }


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    channel: str = Query("default", description="订阅频道: default/alerts/state"),
):
    """WebSocket实时推送"""
    await ws_manager.connect(websocket, channel)
    try:
        await websocket.send_json({
            "type": "connected",
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
        })
        
        await websocket.send_json({
            "channel": "state_update",
            "data": service.get_state()
        })
        
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")
                
                if msg_type == "get_state":
                    await websocket.send_json({
                        "channel": "state_update",
                        "data": service.get_state()
                    })
                elif msg_type == "get_thickness":
                    await websocket.send_json({
                        "channel": "thickness_viz",
                        "data": service.get_thickness_visualization()
                    })
                elif msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
            except json.JSONDecodeError:
                pass
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] 连接异常: {e}")
        ws_manager.disconnect(websocket)


@app.get("/")
async def read_root():
    """服务首页 - 前端页面"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "name": "金箔锻制工艺仿真系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend_warning": "前端页面未找到，请启动前端服务器或构建",
    }


if os.path.exists(FRONTEND_DIR):
    try:
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("  金箔锻制工艺仿真与厚度均匀性分析系统")
    print("  南京金箔锻制工艺数字化研究平台")
    print("="*60)
    print(f"  API 文档: http://localhost:8000/docs")
    print(f"  WebSocket: ws://localhost:8000/ws")
    print(f"  InfluxDB:  {INFLUXDB_URL}")
    print(f"  破裂阈值: {FRACTURE_THRESHOLD_UM}μm")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
