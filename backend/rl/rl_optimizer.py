"""
强化学习优化模块 - 基于厚度分布反馈优化锤击路径和力度
实现策略梯度 + 启发式策略
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.physics_model import (
    GoldFoilPhysicsModel,
    HammerParameters,
)


class ActionType(Enum):
    HEURISTIC = "heuristic"
    RANDOM = "random"
    POLICY_GRADIENT = "policy_gradient"
    Q_LEARNING = "q_learning"


@dataclass
class RLConfig:
    """强化学习配置"""
    grid_size: int = 8
    force_levels: int = 5
    min_force: float = 300.0
    max_force: float = 1500.0
    learning_rate: float = 0.01
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    uniformity_weight: float = 0.6
    thickness_weight: float = 0.3
    fracture_penalty_weight: float = 10.0
    target_thickness_um: float = 0.5


class HammeringPolicy:
    """
    锤击策略 - 整合启发式规则 + 学习参数
    
    状态空间: 厚度偏差图
    动作空间: 锤击位置 (discretized grid) + 力度等级
    """
    
    def __init__(self, foil_size_mm: float, config: Optional[RLConfig] = None):
        self.foil_size_mm = foil_size_mm
        self.config = config or RLConfig()
        self.epsilon = self.config.epsilon_start
        
        n_positions = self.config.grid_size ** 2
        n_actions = n_positions * self.config.force_levels
        self.q_table = np.zeros(
            (self.config.grid_size, self.config.grid_size, self.config.force_levels)
        )
        self.policy_network = np.random.normal(
            0, 0.01,
            (self.config.grid_size, self.config.grid_size, self.config.force_levels)
        )
        self.baseline = 0.0
        self.rewards_history: List[float] = []
        self.action_counts = np.zeros_like(self.q_table)
    
    def _discretize_position(
        self,
        x_mm: float,
        y_mm: float
    ) -> Tuple[int, int]:
        """将连续位置转换为网格索引"""
        half = self.foil_size_mm / 2
        gx = int((x_mm + half) / self.foil_size_mm * self.config.grid_size)
        gy = int((y_mm + half) / self.foil_size_mm * self.config.grid_size)
        gx = np.clip(gx, 0, self.config.grid_size - 1)
        gy = np.clip(gy, 0, self.config.grid_size - 1)
        return gx, gy
    
    def _undiscretize_position(
        self,
        gx: int,
        gy: int
    ) -> Tuple[float, float]:
        """将网格索引转换为连续位置"""
        cell_size = self.foil_size_mm / self.config.grid_size
        x_mm = (gx + 0.5) * cell_size - self.foil_size_mm / 2
        y_mm = (gy + 0.5) * cell_size - self.foil_size_mm / 2
        return x_mm, y_mm
    
    def _force_level_to_value(self, level: int) -> float:
        """力度等级转实际力度值"""
        level = np.clip(level, 0, self.config.force_levels - 1)
        force_range = self.config.max_force - self.config.min_force
        return self.config.min_force + (level / (self.config.force_levels - 1)) * force_range
    
    def _select_action_heuristic(
        self,
        thickness_matrix: np.ndarray,
        temperature: np.ndarray
    ) -> Tuple[int, int, int]:
        """
        基于启发式规则选择动作 - 厚的地方打重锤
        
        策略:
        1. 找到最厚的区域
        2. 厚度偏差越大，力度越大
        """
        h_mean = thickness_matrix.mean()
        h_deviation = thickness_matrix - h_mean
        
        downsampled = self._downsample(h_deviation)
        
        best_gx, best_gy = np.unravel_index(
            np.argmax(downsampled), downsampled.shape
        )
        
        max_dev = downsampled.max()
        avg_dev = np.abs(downsampled).mean()
        if avg_dev > 0:
            intensity_ratio = np.clip(max_dev / (avg_dev + 1e-8), 0, 3)
        else:
            intensity_ratio = 1.0
        
        force_level = int(np.clip(
            (intensity_ratio / 3.0) * (self.config.force_levels - 1)))
        force_level = np.clip(force_level, 0, self.config.force_levels - 1)
        
        return best_gx, best_gy, force_level
    
    def _downsample(self, matrix: np.ndarray) -> np.ndarray:
        """下采样厚度矩阵到策略网格大小"""
        h, w = matrix.shape
        block_h = h // self.config.grid_size
        block_w = w // self.config.grid_size
        
        result = np.zeros((self.config.grid_size, self.config.grid_size))
        
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                start_h = i * block_h
                end_h = (i + 1) * block_h
                start_w = j * block_w
                end_w = (j + 1) * block_w
                result[i, j] = matrix[start_h:end_h, start_w:end_w].mean()
        
        return result
    
    def select_action(
        self,
        thickness_matrix: np.ndarray,
        temperature: np.ndarray,
        mode: ActionType = ActionType.HEURISTIC,
        use_epsilon_greedy: bool = True
    ) -> HammerParameters:
        """
        选择锤击动作
        
        Args:
            thickness_matrix: 当前厚度矩阵
            temperature: 温度分布
            mode: 动作选择模式
            use_epsilon_greedy: 是否使用epsilon贪心
        """
        if use_epsilon_greedy and np.random.random() < self.epsilon:
            gx = np.random.randint(0, self.config.grid_size)
            gy = np.random.randint(0, self.config.grid_size)
            force_level = np.random.randint(0, self.config.force_levels)
        else:
            if mode == ActionType.HEURISTIC:
                gx, gy, force_level = self._select_action_heuristic(
                    thickness_matrix, temperature
                )
            elif mode == ActionType.Q_LEARNING:
                state_repr = self._downsample(thickness_matrix - thickness_matrix.mean())
                q_values = self.q_table + self.policy_network
                flat_idx = np.argmax(q_values.reshape(-1))
                gx, gy, force_level = np.unravel_index(
                    flat_idx, q_values.shape
                )
            else:
                gx, gy, force_level = self._select_action_heuristic(
                    thickness_matrix, temperature
                )
        
        x_mm, y_mm = self._undiscretize_position(gx, gy)
        force = self._force_level_to_value(force_level)
        
        return HammerParameters(
            force=force,
            position=(x_mm, y_mm),
            radius_mm=15.0,
        )
    
    def compute_reward(
        self,
        prev_metrics: dict,
        curr_metrics: dict,
        fracture_risk: dict,
        target_thickness_um: Optional[float] = None
    ) -> float:
        """
        计算奖励函数
        
        奖励构成:
        1. 均匀性改善奖励 (CV降低)
        2. 厚度目标达成奖励
        3. 破裂风险惩罚
        """
        target = target_thickness_um or self.config.target_thickness_um
        
        uniformity_prev = prev_metrics.get("coefficient_of_variation", 0.1)
        uniformity_curr = curr_metrics["coefficient_of_variation"]
        
        uniformity_reward = (uniformity_prev - uniformity_curr) * 100.0
        
        thickness_curr = curr_metrics["mean_thickness_um"]
        thickness_error = abs(thickness_curr - target) / target
        thickness_reward = -thickness_error * 50.0
        
        fracture_penalty = 0.0
        if fracture_risk["risk_level"] != "none":
            risk_scores = {"low": 1, "medium": 3, "high": 10}
            fracture_penalty = -risk_scores.get(
                fracture_risk["risk_level"], 5
            ) * self.config.fracture_penalty_weight
        
        min_thick = curr_metrics["min_thickness_um"]
        if min_thick < 0.1:
            fracture_penalty -= 50.0 * (0.1 - min_thick) / 0.1
        
        total_reward = (
            self.config.uniformity_weight * uniformity_reward
            + self.config.thickness_weight * thickness_reward
            + fracture_penalty
        )
        
        return float(total_reward)
    
    def update(
        self,
        state: np.ndarray,
        action: HammerParameters,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ):
        """
        更新策略参数 (简化的策略梯度更新
        """
        gx, gy = self._discretize_position(*action.position)
        
        force_val = action.force
        force_norm = (force_val - self.config.min_force) / (self.config.max_force - self.config.min_force)
        force_level = int(force_norm * (self.config.force_levels - 1))
        force_level = np.clip(force_level, 0, self.config.force_levels - 1)
        
        self.action_counts[gx, gy, force_level] += 1
        
        advantage = reward - self.baseline
        self.baseline += 0.01 * advantage
        
        lr = self.config.learning_rate
        self.policy_network[gx, gy, force_level] += lr * advantage
        
        state_feat = self._downsample(state - state.mean())
        next_feat = self._downsample(next_state - next_state.mean())
        
        td_target = reward + (0 if done else self.config.gamma * np.max(self.q_table[gx, gy, :]))
        td_error = td_target - self.q_table[gx, gy, force_level]
        self.q_table[gx, gy, force_level] += lr * td_error
        
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        
        self.rewards_history.append(reward)
    
    def get_policy_stats(self) -> dict:
        """获取策略统计信息"""
        return {
            "epsilon": float(self.epsilon),
            "total_actions_taken": int(self.action_counts.sum()),
            "avg_reward": float(np.mean(self.rewards_history[-100:]) if self.rewards_history else 0.0),
            "baseline": float(self.baseline),
            "exploration_rate": float(self.epsilon),
        }


class RLSession:
    """强化学习会话管理"""
    
    def __init__(
        self,
        physics_model: GoldFoilPhysicsModel,
        config: Optional[RLConfig] = None
    ):
        self.physics = physics_model
        self.policy = HammeringPolicy(
            foil_size_mm=physics_model.foil_size_mm,
            config=config
        )
        self.prev_metrics = None
        self.prev_state = None
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.step_count = 0
    
    def step(
        self,
        mode: ActionType = ActionType.HEURISTIC
    ) -> Tuple[HammerParameters, dict, float, dict]:
        """
        执行一步强化学习
        
        Returns:
            (action, strike_result, reward, fracture_risk)
        """
        current_state = self.physics.thickness_um.copy()
        current_metrics = self.physics.get_uniformity_metrics()
        
        if self.prev_metrics is None:
            self.prev_metrics = current_metrics
            self.prev_state = current_state
        
        action = self.policy.select_action(
            thickness_matrix=current_state,
            temperature=self.physics.temperature_c,
            mode=mode,
        )
        
        strike_result = self.physics.apply_hammer_strike(action)
        
        new_metrics = self.physics.get_uniformity_metrics()
        fracture_risk = self.physics.check_fracture_risk()
        
        reward = self.policy.compute_reward(
            prev_metrics=self.prev_metrics,
            curr_metrics=new_metrics,
            fracture_risk=fracture_risk,
        )
        
        self.policy.update(
            state=self.prev_state,
            action=action,
            reward=reward,
            next_state=current_state,
        )
        
        self.prev_metrics = new_metrics
        self.prev_state = current_state
        self.current_episode_reward += reward
        self.step_count += 1
        
        return action, strike_result, reward, fracture_risk
    
    def run_episode(
        self,
        max_steps: int = 50,
        mode: ActionType = ActionType.HEURISTIC
    ) -> dict:
        """运行完整的一集训练"""
        self.physics.reset()
        self.prev_metrics = None
        self.prev_state = None
        self.current_episode_reward = 0.0
        
        actions_taken = []
        rewards_per_step = []
        
        for step in range(max_steps):
            action, strike_result, reward, fracture_risk = self.step(mode=mode)
            actions_taken.append({
                "step": step,
                "action": {
                    "force_N": action.force,
                    "position_mm": list(action.position),
                },
                "reward": reward,
                "avg_thickness_um": strike_result["avg_thickness_um"],
                "cv": strike_result["thickness_std_um"] / strike_result["avg_thickness_um"] if strike_result["avg_thickness_um"] > 0 else 0,
            })
            rewards_per_step.append(reward)
            
            if strike_result["avg_thickness_um"] < 0.15:
                break
        
        self.episode_rewards.append(self.current_episode_reward)
        
        final_metrics = self.physics.get_uniformity_metrics()
        final_risk = self.physics.check_fracture_risk()
        
        return {
            "steps_completed": len(actions_taken),
            "total_reward": self.current_episode_reward,
            "avg_reward_per_step": float(np.mean(rewards_per_step)) if rewards_per_step else 0.0,
            "final_metrics": final_metrics,
            "final_fracture_risk": final_risk,
            "actions_taken": actions_taken,
            "policy_stats": self.policy.get_policy_stats(),
        }
