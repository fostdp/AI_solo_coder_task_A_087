"""
Alarm & WebSocket - 告警评估与WebSocket推送模块

职责:
  - 评估厚度破裂风险
  - 管理WebSocket连接
  - 订阅系统消息并推送到前端
  - 告警历史记录

消息流:
  thickness:updated  →  [风险评估]  →  alarm:triggered (如果有风险)
                                 →  WS推送 (alerts频道)
  strike:result      →  [状态同步]  →  WS推送 (state_update频道)
  system:event       →  [事件广播]  →  WS推送
"""
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

from modules.common import RedisBus, REDIS_CHANNELS

try:
    from fastapi import WebSocket
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


FRACTURE_THRESHOLD_UM = 0.1


class AlarmEvaluator:
    """破裂风险评估器"""

    def __init__(self, threshold_um: float = FRACTURE_THRESHOLD_UM):
        self.threshold_um = threshold_um
        self.alert_history: List[dict] = []
        self._last_risk_level = "none"
        self._consecutive_high_risk = 0
        self._lock = threading.Lock()

    def evaluate(self, thickness_dist: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        评估破裂风险，如果风险等级变化或有新告警则返回告警对象
        返回: alert dict 或 None
        """
        metrics = thickness_dist.get("metrics", {})
        min_thickness = metrics.get("min_thickness_um", 999.0)

        with self._lock:
            below = thickness_dist.get("grid_size", 0)
            if "thickness_matrix_um" in thickness_dist:
                import numpy as np
                h = np.array(thickness_dist["thickness_matrix_um"])
                risk_count = int(np.sum(h < self.threshold_um))
                risk_fraction = risk_count / h.size
            else:
                risk_count = 0
                risk_fraction = 0.0

            if min_thickness < self.threshold_um:
                if risk_fraction < 0.01:
                    risk_level = "low"
                elif risk_fraction < 0.05:
                    risk_level = "medium"
                else:
                    risk_level = "high"
            else:
                risk_level = "none"

            risk_data = {
                "threshold_um": self.threshold_um,
                "risk_level": risk_level,
                "risk_count": risk_count,
                "risk_fraction": risk_fraction,
                "min_thickness_um": min_thickness,
            }

            if risk_level != self._last_risk_level and risk_level != "none":
                alert = {
                    "type": "fracture_warning",
                    "level": risk_level,
                    "message": f"厚度低于{self.threshold_um}μm，破裂风险：{risk_level.upper()}",
                    "risk": risk_data,
                    "timestamp": datetime.now().isoformat(),
                }
                self.alert_history.append(alert)
                if len(self.alert_history) > 500:
                    self.alert_history = self.alert_history[-500:]
                self._last_risk_level = risk_level
                return alert
            elif risk_level == "none" and self._last_risk_level != "none":
                self._last_risk_level = "none"
                self._consecutive_high_risk = 0
                return None
            elif risk_level == "high":
                self._consecutive_high_risk += 1
                if self._consecutive_high_risk % 10 == 1:
                    alert = {
                        "type": "fracture_warning",
                        "level": "high",
                        "message": f"持续高破裂风险，最薄: {min_thickness:.4f}μm",
                        "risk": risk_data,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.alert_history.append(alert)
                    return alert
            return None

    def get_alert_history(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return list(self.alert_history[-limit:])

    def reset(self):
        with self._lock:
            self.alert_history = []
            self._last_risk_level = "none"
            self._consecutive_high_risk = 0


class WSConnectionManager:
    """WebSocket连接管理器 - 支持频道订阅"""

    def __init__(self):
        self.active_connections: Dict[str, List[Any]] = {}
        self.all_connections: List[Any] = []
        self._lock = threading.Lock()

    async def connect(self, websocket, channel: str = "default"):
        await websocket.accept()
        with self._lock:
            self.all_connections.append(websocket)
            if channel not in self.active_connections:
                self.active_connections[channel] = []
            self.active_connections[channel].append(websocket)

    def disconnect(self, websocket):
        with self._lock:
            if websocket in self.all_connections:
                self.all_connections.remove(websocket)
            for channel in self.active_connections:
                if websocket in self.active_connections[channel]:
                    self.active_connections[channel].remove(websocket)

    async def broadcast(self, message: dict, channel: str = None):
        if channel:
            connections = list(self.active_connections.get(channel, []))
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

    def get_connection_count(self) -> int:
        with self._lock:
            return len(self.all_connections)


class AlarmWsService:
    """
    告警与WebSocket整合服务
    
    订阅Redis消息 → 评估告警 → WebSocket推送
    """

    def __init__(self, bus: RedisBus, threshold_um: float = FRACTURE_THRESHOLD_UM):
        self.bus = bus
        self.evaluator = AlarmEvaluator(threshold_um)
        self.ws_manager = WSConnectionManager()

        self.bus.subscribe(REDIS_CHANNELS["thickness_updated"], self._on_thickness_updated)
        self.bus.subscribe(REDIS_CHANNELS["alarm_triggered"], self._on_alarm_triggered)
        self.bus.subscribe(REDIS_CHANNELS["strike_result"], self._on_strike_result)
        self.bus.subscribe(REDIS_CHANNELS["system_event"], self._on_system_event)

    def _on_thickness_updated(self, data: Dict[str, Any]):
        thickness_dist = data.get("thickness_distribution", {})
        alert = self.evaluator.evaluate(thickness_dist)
        if alert:
            self.bus.publish(REDIS_CHANNELS["alarm_triggered"], {
                **alert,
                "foil_id": data.get("foil_id"),
                "session_id": data.get("session_id"),
            })

    def _on_alarm_triggered(self, data: Dict[str, Any]):
        pass

    def _on_strike_result(self, data: Dict[str, Any]):
        pass

    def _on_system_event(self, data: Dict[str, Any]):
        event_type = data.get("type")
        if event_type == "reset":
            self.evaluator.reset()

    def push_state_update(self, state_data: Dict[str, Any]):
        """主动推送状态更新 (由API层调用，异步await)"""
        pass

    async def handle_websocket(self, websocket, channel: str = "default",
                            initial_state: Dict[str, Any] = None):
        """
        处理单个WebSocket连接生命周期
        由 FastAPI endpoint 调用
        """
        await self.ws_manager.connect(websocket, channel)

        try:
            await websocket.send_json({
                "type": "connected",
                "channel": channel,
                "timestamp": datetime.now().isoformat(),
            })

            if initial_state:
                await websocket.send_json({
                    "channel": "state_update",
                    "data": initial_state,
                })

            while True:
                try:
                    data = await websocket.receive_text()
                    import json
                    msg = json.loads(data)
                    msg_type = msg.get("type", "")

                    if msg_type == "get_state":
                        if initial_state:
                            await websocket.send_json({
                                "channel": "state_update",
                                "data": initial_state,
                            })
                    elif msg_type == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat(),
                        })
                except Exception:
                    break

        except Exception:
            pass
        finally:
            self.ws_manager.disconnect(websocket)

    def get_alert_history(self, limit: int = 50) -> List[dict]:
        return self.evaluator.get_alert_history(limit)

    def connection_count(self) -> int:
        return self.ws_manager.get_connection_count()
