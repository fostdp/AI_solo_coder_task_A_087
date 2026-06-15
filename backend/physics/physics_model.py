"""
金箔锻制工艺模型 - 基于塑性力学和乌兹铁匠经验
模拟金箔在反复锤击下的延展和厚度变化
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class MaterialProperties:
    """金的材料力学参数"""
    youngs_modulus: float = 79.0
    poisson_ratio: float = 0.42
    yield_strength: float = 120.0
    ultimate_strength: float = 210.0
    density: float = 19300.0
    initial_thickness_um: float = 500.0
    work_hardening_coeff: float = 0.45
    work_hardening_exp: float = 0.35
    recrystallization_temp: float = 200.0
    melting_point: float = 1064.0


@dataclass
class HammerParameters:
    """锤击参数"""
    force: float = 500.0
    position: Tuple[float, float] = (0.0, 0.0)
    radius_mm: float = 15.0
    strike_duration_ms: float = 50.0


class GoldFoilPhysicsModel:
    """
    金箔锻制物理模型
    
    核心方程:
    1. 塑性应变: ε_p = σ_y / E * (σ/σ_y)^n  (Ludwik硬化法则)
    2. 体积不变: h * A = constant
    3. 延展率: λ = sqrt(A_new / A_old)
    4. 厚度分布: 锤击点产生高斯形变核
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        foil_size_mm: float = 150.0,
        material: Optional[MaterialProperties] = None,
    ):
        self.grid_size = grid_size
        self.foil_size_mm = foil_size_mm
        self.material = material or MaterialProperties()
        self.dx = foil_size_mm / grid_size
        self.dy = foil_size_mm / grid_size
        
        self.thickness_um = np.full(
            (grid_size, grid_size),
            self.material.initial_thickness_um,
            dtype=np.float64
        )
        
        self.plastic_strain = np.zeros(
            (grid_size, grid_size),
            dtype=np.float64
        )
        
        self.temperature_c = np.full(
            (grid_size, grid_size),
            25.0,
            dtype=np.float64
        )
        
        self.strike_count = 0
        self.total_elongation = 1.0
        self.hammer_history: List[dict] = []
    
    def _gaussian_kernel(
        self,
        cx: float,
        cy: float,
        sigma: float
    ) -> np.ndarray:
        """生成2D高斯核，用于模拟锤击形变分布"""
        x = np.linspace(-self.foil_size_mm/2, self.foil_size_mm/2, self.grid_size)
        y = np.linspace(-self.foil_size_mm/2, self.foil_size_mm/2, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        kernel = np.exp(-dist_sq / (2 * sigma ** 2))
        return kernel / kernel.max()
    
    def _calc_effective_stress(
        self,
        force: float,
        radius: float,
        local_thickness: float
    ) -> float:
        """计算等效应力 (MPa)
        
        基于锤击压强公式，考虑冲量放大效应和乌兹铁匠经验系数
        σ = K * F / A，K为动载系数 (150-300)
        """
        contact_area_m2 = np.pi * (radius * 1e-3) ** 2
        contact_pressure_pa = force / contact_area_m2
        
        dynamic_load_factor = 180.0
        
        thickness_factor = np.clip(
            self.material.initial_thickness_um / local_thickness,
            1.0,
            5.0
        )
        
        effective_stress_mpa = (contact_pressure_pa * dynamic_load_factor) * 1e-6
        effective_stress_mpa *= (0.7 + 0.3 * thickness_factor)
        
        return effective_stress_mpa
    
    def _calc_plastic_strain_increment(
        self,
        effective_stress: float,
        accumulated_strain: float
    ) -> float:
        """根据Ludwik-Hollomon方程计算塑性应变增量
        
        σ = σ_y * (1 + (ε_p / ε_y))^n
        """
        if effective_stress <= self.material.yield_strength:
            return 0.0
        
        strain_ratio = effective_stress / self.material.yield_strength
        target_strain = (strain_ratio ** (1.0 / self.material.work_hardening_exp) - 1.0)
        target_strain *= self.material.work_hardening_coeff
        
        strain_increment = max(0.0, target_strain - accumulated_strain)
        return strain_increment
    
    def _calc_thickness_reduction(
        self,
        plastic_strain_inc: float,
        temp_c: float
    ) -> float:
        """根据塑性应变和温度计算厚度减薄率"""
        temp_factor = 1.0
        
        if temp_c >= self.material.recrystallization_temp:
            temp_ratio = (temp_c - self.material.recrystallization_temp) / \
                         (self.material.melting_point - self.material.recrystallization_temp)
            temp_factor = 1.0 + 0.8 * np.clip(temp_ratio, 0.0, 1.0)
        
        reduction = 1.0 - np.exp(-plastic_strain_inc * temp_factor * 0.7)
        return reduction
    
    def apply_hammer_strike(
        self,
        hammer: HammerParameters,
        ambient_temp_c: float = 25.0,
        enable_work_hardening: bool = True
    ) -> dict:
        """
        执行一次锤击，更新厚度分布、应变和温度
        
        返回:
            dict: 包含锤击后状态信息的字典
        """
        cx_mm, cy_mm = hammer.position
        sigma = hammer.radius_mm * 0.8
        
        kernel = self._gaussian_kernel(cx_mm, cy_mm, sigma)
        
        avg_stress = self._calc_effective_stress(
            hammer.force,
            hammer.radius_mm,
            self.thickness_um.mean()
        )
        
        temperature_effect = np.exp(
            -(self.temperature_c - ambient_temp_c) / 300.0
        )
        
        strain_increments = np.zeros_like(self.thickness_um)
        thickness_reductions = np.zeros_like(self.thickness_um)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                local_k = kernel[i, j]
                if local_k < 0.05:
                    continue
                
                local_stress = avg_stress * local_k
                local_strain_inc = self._calc_plastic_strain_increment(
                    local_stress,
                    self.plastic_strain[i, j]
                )
                local_strain_inc *= temperature_effect[i, j]
                
                strain_increments[i, j] = local_strain_inc
                
                thickness_red = self._calc_thickness_reduction(
                    local_strain_inc,
                    self.temperature_c[i, j]
                )
                thickness_reductions[i, j] = thickness_red
        
        new_thickness = self.thickness_um * (1.0 - thickness_reductions)
        
        initial_volume = (self.thickness_um.sum() * self.dx * self.dy)
        new_volume = (new_thickness.sum() * self.dx * self.dy)
        
        volume_correction = initial_volume / new_volume if new_volume > 0 else 1.0
        new_thickness *= volume_correction
        
        new_thickness = np.clip(new_thickness, 0.01, self.material.initial_thickness_um)
        
        self.plastic_strain += strain_increments
        
        if enable_work_hardening:
            pass
        
        heat_generated = thickness_reductions * 150.0
        self.temperature_c = self.temperature_c * 0.95 + (ambient_temp_c + heat_generated) * 0.05
        
        old_area = (self.foil_size_mm) ** 2
        avg_thickness_old = self.thickness_um.mean()
        avg_thickness_new = new_thickness.mean()
        new_area = old_area * (avg_thickness_old / avg_thickness_new)
        
        strike_elongation = np.sqrt(new_area / old_area)
        self.total_elongation *= strike_elongation
        
        self.thickness_um = new_thickness
        self.strike_count += 1
        
        record = {
            "strike_num": self.strike_count,
            "hammer_force_N": hammer.force,
            "hammer_position": (cx_mm, cy_mm),
            "hammer_radius_mm": hammer.radius_mm,
            "avg_thickness_um": float(self.thickness_um.mean()),
            "min_thickness_um": float(self.thickness_um.min()),
            "max_thickness_um": float(self.thickness_um.max()),
            "thickness_std_um": float(self.thickness_um.std()),
            "elongation_rate": float(strike_elongation),
            "total_elongation": float(self.total_elongation),
            "avg_plastic_strain": float(self.plastic_strain.mean()),
            "avg_temperature_c": float(self.temperature_c.mean()),
        }
        self.hammer_history.append(record)
        
        return record
    
    def apply_annealing(
        self,
        temp_c: float = 400.0,
        duration_min: float = 10.0
    ) -> dict:
        """
        模拟退火处理 - 消除加工硬化，恢复塑性
        """
        if temp_c < self.material.recrystallization_temp:
            return {
                "message": "温度低于再结晶温度，退火效果不明显",
                "temp_c": temp_c,
                "residual_strain_ratio": 1.0
            }
        
        temp_ratio = (temp_c - self.material.recrystallization_temp) / \
                     (self.material.melting_point - self.material.recrystallization_temp)
        temp_ratio = np.clip(temp_ratio, 0.0, 1.0)
        
        recrystallization_fraction = 1.0 - np.exp(
            -0.1 * duration_min * temp_ratio ** 2
        )
        
        self.plastic_strain *= (1.0 - recrystallization_fraction * 0.95)
        
        self.temperature_c = np.full_like(self.temperature_c, temp_c)
        
        return {
            "message": f"退火完成，再结晶率: {recrystallization_fraction*100:.1f}%",
            "temp_c": temp_c,
            "duration_min": duration_min,
            "recrystallization_fraction": float(recrystallization_fraction),
            "residual_strain_ratio": float(1.0 - recrystallization_fraction * 0.95)
        }
    
    def get_uniformity_metrics(self) -> dict:
        """
        计算厚度均匀性指标
        """
        h = self.thickness_um
        h_mean = h.mean()
        h_std = h.std()
        h_min = h.min()
        h_max = h.max()
        
        cv = h_std / h_mean if h_mean > 0 else 0
        
        within_5pct = np.sum(np.abs(h - h_mean) <= 0.05 * h_mean) / h.size
        within_10pct = np.sum(np.abs(h - h_mean) <= 0.10 * h_mean) / h.size
        
        diff_central = np.abs(h[self.grid_size//2, self.grid_size//2] - h_mean) / h_mean
        diff_edge = np.abs(h[0, 0] - h_mean) / h_mean
        
        return {
            "mean_thickness_um": float(h_mean),
            "std_thickness_um": float(h_std),
            "min_thickness_um": float(h_min),
            "max_thickness_um": float(h_max),
            "coefficient_of_variation": float(cv),
            "uniformity_within_5pct": float(within_5pct),
            "uniformity_within_10pct": float(within_10pct),
            "center_deviation_ratio": float(diff_central),
            "edge_deviation_ratio": float(diff_edge),
            "range_ratio": float((h_max - h_min) / h_mean) if h_mean > 0 else 0,
        }
    
    def check_fracture_risk(self, threshold_um: float = 0.1) -> dict:
        """
        检查破裂风险 - 厚度低于阈值触发预警
        """
        below_threshold = self.thickness_um < threshold_um
        risk_count = int(np.sum(below_threshold))
        risk_fraction = risk_count / self.thickness_um.size
        
        risk_level = "none"
        if risk_fraction > 0:
            if risk_fraction < 0.01:
                risk_level = "low"
            elif risk_fraction < 0.05:
                risk_level = "medium"
            else:
                risk_level = "high"
        
        positions = []
        if risk_count > 0:
            coords = np.where(below_threshold)
            for i in range(min(risk_count, 10)):
                px_mm = (coords[1][i] / self.grid_size - 0.5) * self.foil_size_mm
                py_mm = (coords[0][i] / self.grid_size - 0.5) * self.foil_size_mm
                positions.append({
                    "x_mm": float(px_mm),
                    "y_mm": float(py_mm),
                    "thickness_um": float(self.thickness_um[coords[0][i], coords[1][i]])
                })
        
        return {
            "threshold_um": threshold_um,
            "risk_level": risk_level,
            "risk_count": risk_count,
            "risk_fraction": float(risk_fraction),
            "risk_positions": positions,
            "min_thickness_um": float(self.thickness_um.min())
        }
    
    def get_thickness_distribution(self) -> dict:
        """获取厚度分布数据用于可视化"""
        h = self.thickness_um
        h_flat = h.flatten()
        
        histogram, bin_edges = np.histogram(
            h_flat,
            bins=32,
            range=(h_flat.min(), h_flat.max())
        )
        
        return {
            "grid_size": self.grid_size,
            "foil_size_mm": self.foil_size_mm,
            "thickness_matrix_um": self.thickness_um.tolist(),
            "histogram": histogram.tolist(),
            "bin_edges": bin_edges.tolist(),
            "metrics": self.get_uniformity_metrics()
        }
    
    def reset(self):
        """重置金箔到初始状态"""
        self.thickness_um = np.full(
            (self.grid_size, self.grid_size),
            self.material.initial_thickness_um,
            dtype=np.float64
        )
        self.plastic_strain = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float64
        )
        self.temperature_c = np.full(
            (self.grid_size, self.grid_size),
            25.0,
            dtype=np.float64
        )
        self.strike_count = 0
        self.total_elongation = 1.0
        self.hammer_history = []
