"""
金箔锻制模拟器 - 模拟传感器每分钟上报锤击力度、温度、厚度分布、延展率
写入InfluxDB时序数据库
"""
import sys
import os
import time
import json
import asyncio
from datetime import datetime, timezone
import random
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

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
from influxdb_client.client.exceptions import InfluxDBError


INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "gold-foil-simulation-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "craftsman-research")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "gold-foil-data")

REPORT_INTERVAL_SEC = int(os.getenv("REPORT_INTERVAL", "60"))
FOIL_ID = os.getenv("FOIL_ID", "NF-001")
CRAFTSMAN_ID = os.getenv("CRAFTSMAN_ID", "master_wu")


class GoldFoilSimulator:
    """金箔锻制模拟器 - 整合物理模型、强化学习、传感器数据上报"""
    
    def __init__(
        self,
        foil_id: str = FOIL_ID,
        craftsman_id: str = CRAFTSMAN_ID,
        grid_size: int = 48,
        use_influxdb: bool = True,
    ):
        self.foil_id = foil_id
        self.craftsman_id = craftsman_id
        self.running = False
        self.session_id = f"session-{int(time.time())}"
        
        material = MaterialProperties(
            initial_thickness_um=500.0,
        )
        self.physics = GoldFoilPhysicsModel(
            grid_size=grid_size,
            foil_size_mm=150.0,
            material=material,
        )
        
        rl_config = RLConfig(
            grid_size=8,
            force_levels=5,
            min_force=300.0,
            max_force=1200.0,
            target_thickness_um=0.5,
        )
        self.rl_session = RLSession(
            physics_model=self.physics,
            config=rl_config,
        )
        
        self.use_influxdb = use_influxdb
        self.influx_client = None
        self.write_api = None
        self.query_api = None
        
        if self.use_influxdb:
            self._init_influxdb()
        
        self.strikes_in_current_minute = 0
        self.last_report_time = time.time()
        self.history_log = []
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_influxdb(self):
        """初始化InfluxDB连接"""
        print(f"[SIM] 连接InfluxDB: {INFLUXDB_URL}")
        try:
            self.influx_client = influxdb_client.InfluxDBClient(
                url=INFLUXDB_URL,
                token=INFLUXDB_TOKEN,
                org=INFLUXDB_ORG
            )
            health = self.influx_client.health()
            print(f"[SIM] InfluxDB状态: {health.status}")
            self.write_api = self.influx_client.write_api(
                write_options=SYNCHRONOUS
            )
            self.query_api = self.influx_client.query_api()
        except Exception as e:
            print(f"[WARN] InfluxDB连接失败，将仅使用内存存储: {e}")
            self.use_influxdb = False
    
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        print(f"\n[SIM] 收到信号 {signum}，正在优雅退出...")
        self.running = False
    
    def _generate_sensor_noise(self, base_value: float, noise_pct: float = 0.02) -> float:
        """生成传感器噪声"""
        noise = random.uniform(-noise_pct, noise_pct)
        return base_value * (1.0 + noise)
    
    def _write_to_influxdb(
        self,
        hammer_result: dict,
        thickness_data: dict,
        fracture_risk: dict,
        rl_action=None,
        rl_reward: float = 0.0,
    ):
        """写入数据到InfluxDB"""
        if not self.use_influxdb or self.write_api is None:
            return
        
        now = datetime.now(timezone.utc)
        
        metrics_point = influxdb_client.Point("forging_metrics") \
            .tag("foil_id", self.foil_id) \
            .tag("craftsman", self.craftsman_id) \
            .tag("session_id", self.session_id) \
            .tag("strike_num", str(hammer_result["strike_num"])) \
            .field("hammer_force", self._generate_sensor_noise(hammer_result["hammer_force_N"])) \
            .field("temperature", self._generate_sensor_noise(hammer_result["avg_temperature_c"])) \
            .field("avg_thickness", self._generate_sensor_noise(hammer_result["avg_thickness_um"])) \
            .field("min_thickness", self._generate_sensor_noise(hammer_result["min_thickness_um"])) \
            .field("max_thickness", self._generate_sensor_noise(hammer_result["max_thickness_um"])) \
            .field("thickness_std", self._generate_sensor_noise(hammer_result["thickness_std_um"])) \
            .field("elongation_rate", self._generate_sensor_noise(hammer_result["elongation_rate"])) \
            .field("total_elongation", self._generate_sensor_noise(hammer_result["total_elongation"])) \
            .field("avg_plastic_strain", hammer_result["avg_plastic_strain"]) \
            .field("hammer_pos_x_mm", hammer_result["hammer_position"][0]) \
            .field("hammer_pos_y_mm", hammer_result["hammer_position"][1]) \
            .time(now)
        
        points = [metrics_point]
        
        uniformity = thickness_data["metrics"]
        uniform_point = influxdb_client.Point("uniformity_metrics") \
            .tag("foil_id", self.foil_id) \
            .tag("craftsman", self.craftsman_id) \
            .tag("session_id", self.session_id) \
            .field("coefficient_of_variation", uniformity["coefficient_of_variation"]) \
            .field("uniformity_within_5pct", uniformity["uniformity_within_5pct"]) \
            .field("uniformity_within_10pct", uniformity["uniformity_within_10pct"]) \
            .field("center_deviation", uniformity["center_deviation_ratio"]) \
            .field("edge_deviation", uniformity["edge_deviation_ratio"]) \
            .field("range_ratio", uniformity["range_ratio"]) \
            .time(now)
        points.append(uniform_point)
        
        risk_point = influxdb_client.Point("fracture_risk") \
            .tag("foil_id", self.foil_id) \
            .tag("craftsman", self.craftsman_id) \
            .tag("session_id", self.session_id) \
            .tag("risk_level", fracture_risk["risk_level"]) \
            .field("risk_count", fracture_risk["risk_count"]) \
            .field("risk_fraction", fracture_risk["risk_fraction"]) \
            .field("min_thickness_um", fracture_risk["min_thickness_um"]) \
            .field("threshold_um", fracture_risk["threshold_um"]) \
            .time(now)
        points.append(risk_point)
        
        if rl_action is not None:
            rl_point = influxdb_client.Point("rl_optimization") \
                .tag("foil_id", self.foil_id) \
                .tag("craftsman", self.craftsman_id) \
                .tag("session_id", self.session_id) \
                .field("rl_reward", rl_reward) \
                .field("rl_action_x_mm", rl_action.position[0]) \
                .field("rl_action_y_mm", rl_action.position[1]) \
                .field("rl_action_force_N", rl_action.force) \
                .time(now)
            points.append(rl_point)
        
        try:
            self.write_api.write(
                bucket=INFLUXDB_BUCKET,
                org=INFLUXDB_ORG,
                record=points
            )
        except InfluxDBError as e:
            print(f"[ERROR] InfluxDB写入失败: {e}")
        except Exception as e:
            print(f"[ERROR] 写入异常: {e}")
    
    def _write_thickness_snapshot(self):
        """写入厚度分布快照（稀疏采样以节省空间）"""
        if not self.use_influxdb or self.write_api is None:
            return
        
        now = datetime.now(timezone.utc)
        h = self.physics.thickness_um
        sample_step = max(1, h.shape[0] // 16)
        
        points = []
        for i in range(0, h.shape[0], sample_step):
            for j in range(0, h.shape[1], sample_step):
                x_mm = (j / h.shape[1] - 0.5) * self.physics.foil_size_mm
                y_mm = (i / h.shape[0] - 0.5) * self.physics.foil_size_mm
                
                point = influxdb_client.Point("thickness_snapshot") \
                    .tag("foil_id", self.foil_id) \
                    .tag("session_id", self.session_id) \
                    .tag("grid_x", str(j)) \
                    .tag("grid_y", str(i)) \
                    .field("x_mm", float(x_mm)) \
                    .field("y_mm", float(y_mm)) \
                    .field("thickness_um", float(h[i, j])) \
                    .time(now)
                points.append(point)
        
        if points:
            try:
                self.write_api.write(
                    bucket=INFLUXDB_BUCKET,
                    org=INFLUXDB_ORG,
                    record=points
                )
            except Exception as e:
                pass
    
    def _report_minute_summary(self):
        """每分钟汇总报告"""
        current_time = time.time()
        elapsed = current_time - self.last_report_time
        
        if elapsed >= REPORT_INTERVAL_SEC:
            thickness_data = self.physics.get_thickness_distribution()
            metrics = thickness_data["metrics"]
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "foil_id": self.foil_id,
                "session_id": self.session_id,
                "strikes_in_period": self.strikes_in_current_minute,
                "total_strikes": self.physics.strike_count,
                "avg_thickness_um": metrics["mean_thickness_um"],
                "thickness_cv": metrics["coefficient_of_variation"],
                "uniformity_10pct": metrics["uniformity_within_10pct"],
                "elongation": self.physics.total_elongation,
                "fracture_risk": self.physics.check_fracture_risk()["risk_level"],
                "temperature_c": self.physics.temperature_c.mean(),
            }
            
            print("\n" + "="*60)
            print(f"[REPORT] {summary['timestamp']}  金箔锻制分钟报告")
            print("="*60)
            print(f"  金箔ID: {self.foil_id}  |  会话: {self.session_id}")
            print(f"  本分钟锤击: {self.strikes_in_current_minute}次  |  累计: {summary['total_strikes']}次")
            print(f"  平均厚度: {metrics['mean_thickness_um']:.3f} μm  |  CV: {metrics['coefficient_of_variation']:.4f}")
            print(f"  厚度范围: [{metrics['min_thickness_um']:.3f}, {metrics['max_thickness_um']:.3f}] μm")
            print(f"  ±10%均匀率: {metrics['uniformity_within_10pct']*100:.1f}%")
            print(f"  总延展率: {self.physics.total_elongation:.2f}x")
            print(f"  温度: {self.physics.temperature_c.mean():.1f}°C  |  破裂风险: {summary['fracture_risk'].upper()}")
            print("="*60 + "\n")
            
            self._write_thickness_snapshot()
            self.history_log.append(summary)
            
            self.strikes_in_current_minute = 0
            self.last_report_time = current_time
            
            return summary
        return None
    
    def run_single_strike(self, mode: ActionType = ActionType.HEURISTIC) -> dict:
        """执行单次锤击模拟"""
        action, strike_result, reward, fracture_risk = self.rl_session.step(mode=mode)
        
        thickness_data = self.physics.get_thickness_distribution()
        
        self._write_to_influxdb(
            hammer_result=strike_result,
            thickness_data=thickness_data,
            fracture_risk=fracture_risk,
            rl_action=action,
            rl_reward=reward,
        )
        
        self.strikes_in_current_minute += 1
        
        result = {
            "strike": strike_result,
            "action": {
                "force_N": action.force,
                "position_mm": list(action.position),
                "radius_mm": action.radius_mm,
            },
            "reward": reward,
            "fracture_risk": fracture_risk,
            "metrics": thickness_data["metrics"],
        }
        
        return result
    
    def run(self, num_strikes: int = None, strike_interval: float = 1.0):
        """
        运行模拟器
        
        Args:
            num_strikes: 总锤击次数，None则无限运行
            strike_interval: 每次锤击间隔（秒）
        """
        self.running = True
        strike_count = 0
        
        print(f"[SIM] 金箔锻制模拟器启动")
        print(f"  金箔ID: {self.foil_id}")
        print(f"  工匠: {self.craftsman_id}")
        print(f"  初始厚度: {self.physics.material.initial_thickness_um}μm")
        print(f"  尺寸: {self.physics.foil_size_mm}mm")
        print(f"  网格: {self.physics.grid_size}x{self.physics.grid_size}")
        print(f"  锤击间隔: {strike_interval}s  |  报告间隔: {REPORT_INTERVAL_SEC}s")
        print(f"  InfluxDB: {'启用' if self.use_influxdb else '禁用'}")
        print(f"  目标厚度: 0.5μm")
        print("-" * 60)
        
        anneal_interval = 15
        
        try:
            while self.running:
                if num_strikes is not None and strike_count >= num_strikes:
                    print(f"\n[SIM] 完成 {num_strikes} 次锤击")
                    break
                
                if strike_count > 0 and strike_count % anneal_interval == 0:
                    avg_strain = self.physics.plastic_strain.mean()
                    if avg_strain > 0.15:
                        print(f"[ANNEAL] 加工硬化严重（应变={avg_strain:.3f}），执行退火...")
                        anneal_result = self.physics.apply_annealing(
                            temp_c=450.0, duration_min=5.0
                        )
                        print(f"[ANNEAL] {anneal_result['message']}")
                
                result = self.run_single_strike(mode=ActionType.HEURISTIC)
                
                strike_count += 1
                
                self._report_minute_summary()
                
                fracture_risk = result["fracture_risk"]
                if fracture_risk["risk_level"] in ["medium", "high"]:
                    print(f"[ALERT] 破裂风险预警! 等级={fracture_risk['risk_level'].upper()}, "
                          f"最薄点={fracture_risk['min_thickness_um']:.4f}μm")
                    if fracture_risk["risk_level"] == "high":
                        print("[WARN] 高破裂风险! 暂停锤击，建议退火")
                
                if result["metrics"]["mean_thickness_um"] < 0.15:
                    print(f"\n[SUCCESS] 达到目标厚度范围！最终厚度: "
                          f"{result['metrics']['mean_thickness_um']:.4f}μm")
                    break
                
                time.sleep(strike_interval)
                
        except KeyboardInterrupt:
            print("\n[SIM] 用户中断")
        finally:
            self.running = False
            print(f"\n[SIM] 模拟器停止，共执行 {strike_count} 次锤击")
            
            final_metrics = self.physics.get_uniformity_metrics()
            print(f"  最终厚度: {final_metrics['mean_thickness_um']:.3f}±"
                  f"{final_metrics['std_thickness_um']:.3f} μm")
            print(f"  最终CV: {final_metrics['coefficient_of_variation']:.4f}")
            print(f"  均匀率(±10%): {final_metrics['uniformity_within_10pct']*100:.1f}%")
            print(f"  总延展率: {self.physics.total_elongation:.2f}x")
            
            self.close()
    
    def close(self):
        """清理资源"""
        if self.influx_client:
            self.influx_client.close()
            print("[SIM] InfluxDB连接已关闭")


def run_demo():
    """运行演示模式 - 50次锤击"""
    print("="*60)
    print("  南京金箔锻制工艺仿真系统 - 模拟器演示")
    print("="*60)
    
    sim = GoldFoilSimulator(
        foil_id="NF-DEMO-001",
        craftsman_id="master_wu_demo",
        grid_size=48,
        use_influxdb=True,
    )
    
    sim.run(num_strikes=50, strike_interval=0.5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="金箔锻制工艺模拟器")
    parser.add_argument("--strikes", type=int, default=None, help="总锤击次数")
    parser.add_argument("--interval", type=float, default=1.0, help="锤击间隔（秒）")
    parser.add_argument("--foil-id", type=str, default=FOIL_ID, help="金箔ID")
    parser.add_argument("--craftsman", type=str, default=CRAFTSMAN_ID, help="工匠ID")
    parser.add_argument("--grid-size", type=int, default=48, help="物理网格大小")
    parser.add_argument("--no-influxdb", action="store_true", help="禁用InfluxDB写入")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        sim = GoldFoilSimulator(
            foil_id=args.foil_id,
            craftsman_id=args.craftsman,
            grid_size=args.grid_size,
            use_influxdb=not args.no_influxdb,
        )
        sim.run(num_strikes=args.strikes, strike_interval=args.interval)
