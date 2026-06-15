"""
DTU Receiver - 传感器数据采集与校验模块

职责:
  - 接收传感器上报的原始数据 (锤击力度、温度、位置等)
  - 数据格式校验、范围校验、异常值过滤
  - 校验通过后发布到 hammer:request 频道供塑性仿真消费

消息流:
  REST/sensor_raw  →  [DTU校验]  →  hammer:request (valid)
                      ↓
                  sensor:validation (校验结果事件)
"""
import time
import uuid
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

from modules.common import RedisBus, REDIS_CHANNELS, load_material_config


@dataclass
class ValidationResult:
    valid: bool
    errors: list
    warnings: list


class DtuReceiver:
    """
    DTU数据接收器 - 负责传感器数据采集与校验
    
    支持两种模式:
    1. 订阅 sensor_raw 频道 (异步采集)
    2. 直接调用 validate() 方法 (同步校验)
    """

    def __init__(self, bus: RedisBus):
        self.bus = bus
        self.config = load_material_config()
        self.stats = {
            "total_received": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "last_received": None,
        }

        physics_cfg = self.config.get("physics", {})
        hammer_cfg = self.config.get("hammer", {})
        foil_size = physics_cfg.get("foil_size_mm", 150.0)

        self.validation_rules = {
            "force_N": {"min": 100.0, "max": 3000.0, "required": True},
            "position_x_mm": {"min": -foil_size / 2, "max": foil_size / 2, "required": True},
            "position_y_mm": {"min": -foil_size / 2, "max": foil_size / 2, "required": True},
            "radius_mm": {"min": 5.0, "max": 50.0, "required": False, "default": hammer_cfg.get("default_radius_mm", 15.0)},
            "temperature_c": {"min": -50.0, "max": 1200.0, "required": False, "default": 25.0},
            "sensor_id": {"required": False, "default": "dtu-001"},
        }

        self.bus.subscribe(REDIS_CHANNELS["sensor_raw"], self._on_sensor_raw)

    def validate(self, data: Dict[str, Any]) -> Tuple[ValidationResult, Dict[str, Any]]:
        """
        校验传感器数据，返回校验结果和标准化后的数据
        """
        errors = []
        warnings = []
        cleaned = {}

        for field, rules in self.validation_rules.items():
            value = data.get(field)

            if value is None:
                if rules.get("required", False):
                    errors.append(f"缺少必填字段: {field}")
                    continue
                else:
                    value = rules.get("default")

            if "min" in rules and value < rules["min"]:
                errors.append(f"{field}={value} 超出下限 {rules['min']}")
                continue

            if "max" in rules and value > rules["max"]:
                errors.append(f"{field}={value} 超出上限 {rules['max']}")
                continue

            if field in ("force_N", "radius_mm") and value < rules["min"] * 1.1:
                warnings.append(f"{field}={value} 接近下限，建议检查传感器")

            cleaned[field] = value

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings), cleaned

    def _on_sensor_raw(self, data: Dict[str, Any]):
        """处理 sensor_raw 频道消息"""
        self.stats["total_received"] += 1
        self.stats["last_received"] = time.time()

        result, cleaned = self.validate(data)

        if not result.valid:
            self.stats["invalid_count"] += 1
            self.bus.publish("sensor:validation", {
                "valid": False,
                "errors": result.errors,
                "original": data,
                "timestamp": time.time(),
            })
            return

        self.stats["valid_count"] += 1

        hammer_msg = {
            "request_id": str(uuid.uuid4()),
            "force": cleaned["force_N"],
            "position": [cleaned["position_x_mm"], cleaned["position_y_mm"]],
            "radius_mm": cleaned["radius_mm"],
            "ambient_temp_c": cleaned.get("temperature_c", 25.0),
            "source": "dtu_sensor",
            "sensor_id": cleaned.get("sensor_id", "dtu-001"),
            "timestamp": time.time(),
        }

        self.bus.publish(REDIS_CHANNELS["hammer_request"], hammer_msg)

        if result.warnings:
            self.bus.publish("sensor:validation", {
                "valid": True,
                "warnings": result.warnings,
                "original": data,
                "timestamp": time.time(),
            })

    def submit_hammer(self, force: float, position: tuple, radius_mm: float = None,
                      source: str = "manual") -> Dict[str, Any]:
        """
        直接提交锤击请求 (供REST API调用)
        返回: {request_id, valid, errors?}
        """
        payload = {
            "force_N": force,
            "position_x_mm": position[0],
            "position_y_mm": position[1],
            "radius_mm": radius_mm or self.config.get("hammer", {}).get("default_radius_mm", 15.0),
        }

        result, cleaned = self.validate(payload)
        if not result.valid:
            return {"valid": False, "errors": result.errors}

        request_id = str(uuid.uuid4())
        hammer_msg = {
            "request_id": request_id,
            "force": cleaned["force_N"],
            "position": [cleaned["position_x_mm"], cleaned["position_y_mm"]],
            "radius_mm": cleaned["radius_mm"],
            "ambient_temp_c": 25.0,
            "source": source,
            "timestamp": time.time(),
        }

        self.bus.publish(REDIS_CHANNELS["hammer_request"], hammer_msg)
        self.stats["valid_count"] += 1
        self.stats["total_received"] += 1

        return {"valid": True, "request_id": request_id}

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)

    def reset_stats(self):
        self.stats = {
            "total_received": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "last_received": None,
        }
