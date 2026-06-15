"""
RL Optimizer - 强化学习路径优化模块

职责:
  - 维护锤击策略 (Q-Learning + 策略梯度 + 预训练
  - 订阅 rl:action:request 频道，返回最优锤击动作
  - 订阅 thickness:updated / strike:result 频道，更新策略
  - 支持演示数据预训练 (Behavior Cloning)

消息流:
  rl:action:request  →  [策略选择]  →  rl:action:result
  thickness:updated    →  [策略更新]  →  (内部更新)
  strike:result        →  [策略更新]  →  (内部更新)
"""
import time
import threading
from typing import Dict, Any, Optional, Tuple

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

from modules.common import RedisBus, REDIS_CHANNELS, load_rl_config, load_material_config
from rl.rl_optimizer import (
    RLSession,
    ActionType,
    RLConfig,
    DemoBuffer,
    HammeringPolicy,
)
from physics.physics_model import GoldFoilPhysicsModel, MaterialProperties


class RlOptimizerModule:
    """
    强化学习优化服务
    
    两种运行模式:
    1. 独立模式: 订阅请求，计算动作，发布结果
    2. 集成模式: 直接持有物理模型引用，直接step同步更新
    """

    def __init__(self, bus: RedisBus, physics_model: GoldFoilPhysicsModel = None):
        self.bus = bus
        self.rl_cfg_raw = load_rl_config()
        self.mat_cfg = load_material_config()

        rl_params = self.rl_cfg_raw.get("rl", {})
        reward_params = self.rl_cfg_raw.get("reward", {})
        pretrain_params = self.rl_cfg_raw.get("pretrain", {})

        config = RLConfig(
            grid_size=rl_params.get("grid_size", 8),
            force_levels=rl_params.get("force_levels", 5),
            min_force=rl_params.get("min_force", 300.0),
            max_force=rl_params.get("max_force", 1500.0),
            learning_rate=rl_params.get("learning_rate", 0.01),
            gamma=rl_params.get("gamma", 0.99),
            epsilon_start=rl_params.get("epsilon_start", 1.0),
            epsilon_min=rl_params.get("epsilon_min", 0.05),
            epsilon_decay=rl_params.get("epsilon_decay", 0.995),
            uniformity_weight=reward_params.get("uniformity_weight", 0.6),
            thickness_weight=reward_params.get("thickness_weight", 0.3),
            fracture_penalty_weight=reward_params.get("fracture_penalty_weight", 10.0),
            target_thickness_um=rl_params.get("target_thickness_um", 0.5),
            pretrain_lr=pretrain_params.get("pretrain_lr", 0.05),
            pretrain_epochs=pretrain_params.get("pretrain_epochs", 50),
            pretrain_batch_size=pretrain_params.get("pretrain_batch_size", 64),
        )

        self._lock = threading.Lock()

        physics_cfg = self.mat_cfg.get("physics", {})
        self.foil_size = physics_cfg.get("foil_size_mm", 150.0)

        if physics_model is not None:
            self.physics = physics_model
            self.session = RLSession(physics_model=self.physics, config=config)
        else:
            material = MaterialProperties(**self.mat_cfg.get("material", {}))
            self.physics = GoldFoilPhysicsModel(
                grid_size=physics_cfg.get("default_grid_size", 48),
                foil_size_mm=self.foil_size,
                material=material,
            )
            self.session = RLSession(physics_model=self.physics, config=config)

        self.pretrain_running = False
        self.pretrain_report = None

        self.bus.subscribe(REDIS_CHANNELS["rl_action_request"], self._on_action_request)
        self.bus.subscribe(REDIS_CHANNELS["thickness_updated"], self._on_thickness_updated)
        self.bus.subscribe(REDIS_CHANNELS["system_event"], self._on_system_event)

    def _on_action_request(self, data: Dict[str, Any]):
        """处理动作请求"""
        with self._lock:
            try:
                mode_str = data.get("mode", "heuristic")
                mode = self._parse_action_type(mode_str)

                if mode == ActionType.PRETRAINED and not self.session.policy.is_pretrained:
                    mode = ActionType.HEURISTIC

                action = self.session.policy.select_action(
                    thickness_matrix=self.physics.thickness_um,
                    temperature=self.physics.temperature_c,
                    mode=mode,
                )

                self.bus.publish(REDIS_CHANNELS["rl_action_result"], {
                    "request_id": data.get("request_id"),
                    "action": {
                        "force_N": action.force,
                        "position_mm": list(action.position),
                        "radius_mm": action.radius_mm,
                        "mode": mode_str,
                    },
                    "policy_stats": self.session.policy.get_policy_stats(),
                    "timestamp": time.time(),
                })

            except Exception as e:
                print(f"[ERROR] RL动作计算失败: {e}")
                import traceback
                traceback.print_exc()

    def _on_thickness_updated(self, data: Dict[str, Any]):
        """厚度更新时同步策略 (在线学习)"""
        pass

    def _on_system_event(self, data: Dict[str, Any]):
        event_type = data.get("type")
        if event_type == "reset":
            with self._lock:
                self.physics.reset()
                self.session.prev_metrics = None
                self.session.prev_state = None
                self.session.current_episode_reward = 0.0
                self.session.step_count = 0

    def step_with_physics(self, mode: ActionType = ActionType.HEURISTIC) -> Tuple:
        """
        同步执行一步RL (集成模式)
        返回: (action, strike_result, reward, fracture_risk)
        """
        with self._lock:
            return self.session.step(mode=mode)

    def trigger_pretrain_async(self, num_demos: int = None, steps_per_demo: int = None,
                           pretrain_epochs: int = None) -> Dict[str, Any]:
        """
        后台线程执行预训练
        """
        if self.pretrain_running:
            return {"running": True, "message": "预训练已在进行中"}
        if self.session.policy.is_pretrained:
            return {"running": False, "message": "已完成预训练", "report": self.pretrain_report}

        self.pretrain_running = True

        def worker():
            try:
                print("[RL] 开始后台预训练...")
                params = self.rl_cfg_raw.get("pretrain", {})
                report = self.session.generate_and_pretrain(
                    num_demos=num_demos or params.get("num_demos", 20),
                    steps_per_demo=steps_per_demo or params.get("steps_per_demo", 40),
                    pretrain_epochs=pretrain_epochs or params.get("pretrain_epochs", 50),
                    verbose=True,
                )
                self.pretrain_report = report
                print(f"[RL] 预训练完成! 位置准确率={report['behavior_cloning'].get('final_position_accuracy', 0):.2%}")
                self.bus.publish(REDIS_CHANNELS["system_event"], {
                    "type": "pretrain_complete",
                    "report": report,
                    "timestamp": time.time(),
                })
            except Exception as e:
                print(f"[RL] 预训练异常: {e}")
            finally:
                self.pretrain_running = False

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return {"running": True, "message": "预训练已启动，后台执行中"}

    def get_policy_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = self.session.policy.get_policy_stats()
            stats["pretrain_running"] = self.pretrain_running
            stats["pretrain_report"] = self.pretrain_report
            return stats

    @staticmethod
    def _parse_action_type(mode_str: str) -> ActionType:
        mapping = {
            "heuristic": ActionType.HEURISTIC,
            "rl": ActionType.Q_LEARNING,
            "q_learning": ActionType.Q_LEARNING,
            "pretrained": ActionType.PRETRAINED,
            "policy_gradient": ActionType.POLICY_GRADIENT,
            "random": ActionType.RANDOM,
        }
        return mapping.get(mode_str, ActionType.HEURISTIC)
