"""
Plasticity Simulator - 塑性变形计算模块

职责:
  - 维护金箔物理模型状态 (厚度、应变、温度)
  - 订阅 hammer:request 频道，执行塑性变形计算
  - 执行自适应网格重划
  - 发布 strike:result、thickness:updated、mesh:quality 消息

消息流:
  hammer:request  →  [塑性仿真]  →  strike:result
                                 →  thickness:updated
                                 →  mesh:quality (重划时)
  system:event    →  [状态控制]  →  reset/anneal
"""
import time
import threading
from typing import Dict, Any, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

from modules.common import RedisBus, REDIS_CHANNELS, load_material_config
from physics.physics_model import (
    GoldFoilPhysicsModel,
    HammerParameters,
    MaterialProperties,
    RemeshConfig,
)


class PlasticitySimulator:
    """
    塑性变形计算服务 - 持有物理模型，响应锤击请求
    """

    def __init__(self, bus: RedisBus, grid_size: int = None):
        self.bus = bus
        self.config = load_material_config()
        self._lock = threading.Lock()

        physics_cfg = self.config.get("physics", {})
        mat_cfg = self.config.get("material", {})
        remesh_cfg = self.config.get("remesh", {})

        self.foil_size = physics_cfg.get("foil_size_mm", 150.0)
        initial_grid = grid_size or physics_cfg.get("default_grid_size", 48)

        material = MaterialProperties(**mat_cfg)
        remesh_config = RemeshConfig(**remesh_cfg) if remesh_cfg.get("enable", True) else None

        self.physics = GoldFoilPhysicsModel(
            grid_size=initial_grid,
            foil_size_mm=self.foil_size,
            material=material,
            remesh_config=remesh_config,
        )

        self.foil_id = "NF-LIVE-001"
        self.session_id = f"session-{int(time.time())}"
        self.strike_history = []

        self.bus.subscribe(REDIS_CHANNELS["hammer_request"], self._on_hammer_request)
        self.bus.subscribe(REDIS_CHANNELS["system_event"], self._on_system_event)

    def _on_hammer_request(self, data: Dict[str, Any]):
        """处理锤击请求"""
        with self._lock:
            try:
                hammer = HammerParameters(
                    force=data.get("force", 500.0),
                    position=tuple(data.get("position", [0.0, 0.0])),
                    radius_mm=data.get("radius_mm", 15.0),
                )

                ambient_temp = data.get("ambient_temp_c", 25.0)
                result = self.physics.apply_hammer_strike(hammer, ambient_temp_c=ambient_temp)

                self._record_strike(result, data.get("request_id"), data.get("source"))

                self.bus.publish(REDIS_CHANNELS["strike_result"], {
                    "request_id": data.get("request_id"),
                    "strike": result,
                    "foil_id": self.foil_id,
                    "session_id": self.session_id,
                    "timestamp": time.time(),
                })

                thickness_dist = self.physics.get_thickness_distribution()
                self.bus.publish(REDIS_CHANNELS["thickness_updated"], {
                    "thickness_distribution": thickness_dist,
                    "metrics": thickness_dist["metrics"],
                    "foil_id": self.foil_id,
                    "session_id": self.session_id,
                    "strike_num": result["strike_num"],
                    "timestamp": time.time(),
                })

                if result.get("remesh") and result["remesh"].get("action") != "noop":
                    self.bus.publish(REDIS_CHANNELS["mesh_quality"], {
                        "remesh_event": result["remesh"],
                        "quality_report": self.physics.get_mesh_quality_report(),
                        "timestamp": time.time(),
                    })

            except Exception as e:
                print(f"[ERROR] 塑性仿真处理失败: {e}")
                import traceback
                traceback.print_exc()

    def _on_system_event(self, data: Dict[str, Any]):
        """处理系统事件 (reset/anneal等)"""
        event_type = data.get("type")

        with self._lock:
            if event_type == "reset":
                self.physics.reset()
                self.session_id = f"session-{int(time.time())}"
                self.strike_history = []
                self.bus.publish(REDIS_CHANNELS["system_event"], {
                    "type": "reset_complete",
                    "session_id": self.session_id,
                    "timestamp": time.time(),
                })

            elif event_type == "anneal":
                temp_c = data.get("temperature_c", 400.0)
                duration_min = data.get("duration_min", 10.0)
                result = self.physics.apply_annealing(temp_c, duration_min)
                self.bus.publish(REDIS_CHANNELS["system_event"], {
                    "type": "anneal_complete",
                    "result": result,
                    "metrics": self.physics.get_uniformity_metrics(),
                    "timestamp": time.time(),
                })

    def _record_strike(self, result: Dict[str, Any], request_id: str, source: str):
        record = {
            "request_id": request_id,
            "source": source,
            **result,
        }
        self.strike_history.append(record)
        if len(self.strike_history) > 1000:
            self.strike_history = self.strike_history[-1000:]

    def get_state(self) -> Dict[str, Any]:
        """获取当前仿真状态 (供API层直接调用)"""
        with self._lock:
            thickness_data = self.physics.get_thickness_distribution()
            return {
                "foil_id": self.foil_id,
                "session_id": self.session_id,
                "total_strikes": self.physics.strike_count,
                "total_elongation": self.physics.total_elongation,
                "thickness_distribution": thickness_data,
                "fracture_risk": self.physics.check_fracture_risk(0.1),
                "temperature_c": float(self.physics.temperature_c.mean()),
                "plastic_strain": float(self.physics.plastic_strain.mean()),
                "grid_size": self.physics.grid_size,
            }

    def get_thickness_viz(self) -> Dict[str, Any]:
        """获取厚度可视化数据"""
        with self._lock:
            h = self.physics.thickness_um
            h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)
            return {
                "grid_size": self.physics.grid_size,
                "foil_size_mm": self.foil_size,
                "thickness_um": h.tolist(),
                "normalized": h_norm.tolist(),
                "min_um": float(h.min()),
                "max_um": float(h.max()),
                "mean_um": float(h.mean()),
                "std_um": float(h.std()),
            }

    def get_mesh_quality(self) -> Dict[str, Any]:
        """获取网格质量报告"""
        with self._lock:
            return self.physics.get_mesh_quality_report()

    def apply_strike_direct(self, hammer: HammerParameters) -> Dict[str, Any]:
        """
        同步执行锤击 (不走Redis，供内部直接调用)
        返回: strike结果
        """
        with self._lock:
            result = self.physics.apply_hammer_strike(hammer)
            return result

    def reset(self):
        """重置仿真 (同步)"""
        with self._lock:
            self.physics.reset()
            self.session_id = f"session-{int(time.time())}"
            self.strike_history = []
