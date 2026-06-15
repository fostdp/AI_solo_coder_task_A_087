"""
金箔锻制模拟器 - 可配置化锤击路径与力度
支持多种预设路径、自定义路径、力度分布，并写入InfluxDB

用法:
  # 使用南京乌金打工艺预设
  python scripts/simulator.py --preset nanjing_wujin

  # 螺旋路径 + 重锤力度
  python scripts/simulator.py --path spiral --force heavy --strikes 200

  # 自定义锤击序列 (JSON文件)
  python scripts/simulator.py --custom-path my_path.json
"""
import sys
import os
import time
import json
import math
import random
import signal
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from physics.physics_model import (
    GoldFoilPhysicsModel,
    HammerParameters,
    MaterialProperties,
    RemeshConfig,
)
from rl.rl_optimizer import RLSession, ActionType, RLConfig
from modules.common import load_material_config, load_rl_config

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
API_BASE = os.getenv("API_BASE", None)
FOIL_SIZE_MM = 150.0


FORCE_PRESETS: Dict[str, Dict[str, float]] = {
    "light": {
        "min": 300.0, "max": 600.0, "mean": 450.0, "std": 100.0,
        "radius_mm": 15.0, "anneal_strain": 0.2, "desc": "轻锤(精整阶段)"
    },
    "medium": {
        "min": 600.0, "max": 1200.0, "mean": 900.0, "std": 200.0,
        "radius_mm": 15.0, "anneal_strain": 0.18, "desc": "中锤(延展阶段)"
    },
    "heavy": {
        "min": 1000.0, "max": 1800.0, "mean": 1400.0, "std": 250.0,
        "radius_mm": 18.0, "anneal_strain": 0.15, "desc": "重锤(开坯阶段)"
    },
    "nanjing_wujin": {
        "min": 500.0, "max": 1500.0, "mean": 900.0, "std": 300.0,
        "radius_mm": 16.0, "anneal_strain": 0.18, "desc": "南京乌金打传统力度"
    },
    "variable": {
        "min": 300.0, "max": 1800.0, "mean": 1000.0, "std": 400.0,
        "radius_mm": 15.0, "anneal_strain": 0.18, "desc": "随应力自动调整力度"
    },
}


PATH_PRESETS = {
    "center_out": "由中心向外分层打（南京金箔传统路径）",
    "spiral": "阿基米德螺旋线",
    "grid_scan": "逐行扫描网格化",
    "diagonal": "对角线交叉打",
    "random": "随机厚处优先",
    "heuristic": "强化学习启发式厚处打重锤",
    "rl": "纯强化学习策略",
    "pretrained": "演示预训练策略",
}


@dataclass
class StrikeEntry:
    pos_x: float
    pos_y: float
    force: Optional[float] = None
    radius_mm: Optional[float] = None
    wait_sec: float = 0.0
    label: str = ""


def generate_path_points(
    path_type: str,
    num_points: int,
    foil_size: float = FOIL_SIZE_MM,
) -> List[Tuple[float, float]]:
    R = foil_size / 2 * 0.85
    rng = random.Random(42)

    points = []

    if path_type == "center_out":
        rings = max(1, int(math.sqrt(num_points / 4)))
        total = 0
        for ring in range(rings + 1):
            r = R * (ring / rings) if rings > 0 else 0
            count = max(1, int(2 * math.pi * ring / 2) + (1 if ring == 0 else 0))
            for k in range(count):
                if total >= num_points:
                    break
                theta = 2 * math.pi * k / max(1, count) + rng.uniform(-0.1, 0.1)
                points.append((
                    r * math.cos(theta),
                    r * math.sin(theta),
                ))
                total += 1

    elif path_type == "spiral":
        turns = 4.0
        for t in range(num_points):
            ratio = t / max(1, num_points - 1)
            theta = turns * 2 * math.pi * ratio
            r = R * ratio + rng.uniform(-1.0, 1.0)
            points.append((r * math.cos(theta), r * math.sin(theta)))

    elif path_type == "grid_scan":
        side = int(math.sqrt(num_points)) + 1
        step = (foil_size * 0.85) / max(1, side - 1)
        offset = -foil_size * 0.425
        for i in range(side):
            for j in range(side):
                if len(points) >= num_points:
                    break
                points.append((offset + j * step, offset + i * step))
            if len(points) >= num_points:
                break

    elif path_type == "diagonal":
        per_phase = num_points // 2
        for phase in range(2):
            for t in range(per_phase):
                ratio = t / max(1, per_phase - 1)
                if phase == 0:
                    x = -R * 0.7 + 2 * R * 0.7 * ratio
                    y = R * 0.7 - 2 * R * 0.7 * ratio
                else:
                    x = -R * 0.7 + 2 * R * 0.7 * ratio
                    y = -R * 0.7 + 2 * R * 0.7 * ratio
                points.append((
                    x + rng.uniform(-1.5, 1.5),
                    y + rng.uniform(-1.5, 1.5),
                ))

    elif path_type == "random":
        for _ in range(num_points):
            r = math.sqrt(rng.random()) * R
            theta = rng.random() * 2 * math.pi
            points.append((r * math.cos(theta), r * math.sin(theta)))

    elif path_type in ("heuristic", "rl", "pretrained"):
        points = [(0.0, 0.0)] * num_points

    return points[:num_points]


def generate_force_value(
    preset: str,
    index: int,
    total: int,
    physics: Optional[GoldFoilPhysicsModel] = None,
) -> Tuple[float, float]:
    cfg = FORCE_PRESETS.get(preset, FORCE_PRESETS["medium"])

    if preset == "variable" and physics is not None:
        std_thick = float(physics.thickness_um.std())
        mean_thick = float(physics.thickness_um.mean())
        cv = std_thick / max(mean_thick, 1e-6)
        base_force = cfg["min"] + (cfg["max"] - cfg["min"]) * min(cv / 0.3, 1.0)
    else:
        progress = index / max(1, total - 1)
        taper = 1.0 - 0.4 * progress
        base_force = cfg["mean"] * taper

    noise = rng = random.Random(index + int(time.time() * 1000) % 1000)
    force = base_force + rng.gauss(0, cfg["std"] * 0.3)
    force = max(cfg["min"], min(cfg["max"], force))

    return force, cfg["radius_mm"]


def load_custom_path(file_path: str) -> List[StrikeEntry]:
    if not os.path.exists(file_path):
        print(f"[ERROR] 自定义路径文件不存在: {file_path}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for item in data.get("strikes", data):
        entries.append(StrikeEntry(
            pos_x=float(item.get("pos_x", item.get("x", 0))),
            pos_y=float(item.get("pos_y", item.get("y", 0))),
            force=float(item["force"]) if "force" in item else None,
            radius_mm=float(item.get("radius_mm", 15)) if "radius_mm" in item else None,
            wait_sec=float(item.get("wait_sec", 0)),
            label=item.get("label", ""),
        ))
    print(f"[SIM] 加载自定义锤击序列: {len(entries)} 步, 文件: {file_path}")
    return entries


PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "nanjing_wujin": {
        "description": "南京乌金打传统工艺 - 先重后轻，中心向外，7道退火",
        "path": "center_out",
        "force": "nanjing_wujin",
        "strikes": 500,
        "interval": 0.8,
        "anneal_interval_strikes": 70,
        "anneal_temp": 450,
        "initial_thickness": 1000.0,
    },
    "kaipi": {
        "description": "开坯阶段 - 重锤快速延展",
        "path": "grid_scan",
        "force": "heavy",
        "strikes": 120,
        "interval": 1.2,
        "anneal_interval_strikes": 40,
        "anneal_temp": 400,
        "initial_thickness": 2000.0,
    },
    "yanhou": {
        "description": "延后阶段 - 中锤控制均匀性",
        "path": "spiral",
        "force": "medium",
        "strikes": 300,
        "interval": 0.9,
        "anneal_interval_strikes": 80,
        "anneal_temp": 420,
        "initial_thickness": 800.0,
    },
    "jingzheng": {
        "description": "精整阶段 - 轻锤高频修形",
        "path": "center_out",
        "force": "light",
        "strikes": 800,
        "interval": 0.3,
        "anneal_interval_strikes": 150,
        "anneal_temp": 380,
        "initial_thickness": 50.0,
    },
    "ai_optimized": {
        "description": "AI强化学习优化 - 预训练策略自动路径与力度",
        "path": "pretrained",
        "force": "variable",
        "strikes": 300,
        "interval": 0.5,
        "anneal_interval_strikes": 60,
        "anneal_temp": 420,
        "initial_thickness": 1000.0,
    },
}


class GoldFoilSimulator:
    def __init__(
        self,
        foil_id: str = FOIL_ID,
        craftsman_id: str = CRAFTSMAN_ID,
        grid_size: int = 48,
        use_influxdb: bool = True,
        api_base: Optional[str] = None,
    ):
        self.foil_id = foil_id
        self.craftsman_id = craftsman_id
        self.running = False
        self.session_id = f"session-{int(time.time())}"

        mat_cfg = load_material_config()
        material = MaterialProperties(**mat_cfg.get("material", {}))
        remesh_cfg = RemeshConfig(**mat_cfg.get("remesh", {}))

        self.physics = GoldFoilPhysicsModel(
            grid_size=grid_size,
            foil_size_mm=FOIL_SIZE_MM,
            material=material,
            remesh_config=remesh_cfg,
        )

        rl_cfg = load_rl_config()
        rlp = rl_cfg.get("rl", {})
        rp = rl_cfg.get("reward", {})
        pp = rl_cfg.get("pretrain", {})
        config = RLConfig(
            grid_size=rlp.get("grid_size", 8),
            force_levels=rlp.get("force_levels", 5),
            min_force=rlp.get("min_force", 300),
            max_force=rlp.get("max_force", 1500),
            target_thickness_um=rlp.get("target_thickness_um", 0.5),
            uniformity_weight=rp.get("uniformity_weight", 0.6),
            thickness_weight=rp.get("thickness_weight", 0.3),
            pretrain_epochs=pp.get("pretrain_epochs", 50),
        )
        self.rl_session = RLSession(physics_model=self.physics, config=config)

        self.use_influxdb = use_influxdb
        self.api_base = api_base
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
        print(f"[SIM] 连接InfluxDB: {INFLUXDB_URL}")
        try:
            self.influx_client = influxdb_client.InfluxDBClient(
                url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
            )
            health = self.influx_client.health()
            print(f"[SIM] InfluxDB状态: {health.status}")
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.influx_client.query_api()
        except Exception as e:
            print(f"[WARN] InfluxDB连接失败，仅使用内存存储: {e}")
            self.use_influxdb = False

    def _signal_handler(self, signum, frame):
        print(f"\n[SIM] 收到信号 {signum}，优雅退出...")
        self.running = False

    def _write_points(self, points):
        if not self.use_influxdb or self.write_api is None:
            return
        try:
            self.write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points)
        except Exception as e:
            print(f"[WARN] InfluxDB写入失败: {e}")

    def _persist_step(
        self, strike, fracture_risk, action=None, reward=0.0, path_label: str = ""
    ):
        if not self.use_influxdb:
            return
        t = self.physics.thickness_um
        thickness_data = self.physics.get_thickness_distribution()
        metrics = thickness_data["metrics"]
        now = datetime.now(timezone.utc)

        points = []
        p1 = influxdb_client.Point("forging_metrics") \
            .tag("foil_id", self.foil_id).tag("craftsman", self.craftsman_id) \
            .tag("session_id", self.session_id) \
            .tag("strike_num", str(strike["strike_num"])).tag("path_label", path_label) \
            .field("hammer_force", strike["hammer_force_N"]) \
            .field("temperature", strike["avg_temperature_c"]) \
            .field("avg_thickness", strike["avg_thickness_um"]) \
            .field("min_thickness", strike["min_thickness_um"]) \
            .field("thickness_std", strike["thickness_std_um"]) \
            .field("elongation_rate", strike["elongation_rate"]) \
            .field("total_elongation", strike["total_elongation"]) \
            .field("hammer_pos_x_mm", strike["hammer_position"][0]) \
            .field("hammer_pos_y_mm", strike["hammer_position"][1]) \
            .field("grid_size", strike.get("grid_size", self.physics.grid_size)) \
            .time(now)
        points.append(p1)

        p2 = influxdb_client.Point("uniformity_metrics") \
            .tag("foil_id", self.foil_id).tag("session_id", self.session_id) \
            .field("coefficient_of_variation", metrics["coefficient_of_variation"]) \
            .field("uniformity_within_5pct", metrics["uniformity_within_5pct"]) \
            .field("uniformity_within_10pct", metrics["uniformity_within_10pct"]) \
            .field("range_ratio", metrics["range_ratio"]) \
            .time(now)
        points.append(p2)

        p3 = influxdb_client.Point("fracture_risk") \
            .tag("foil_id", self.foil_id).tag("session_id", self.session_id) \
            .tag("risk_level", fracture_risk["risk_level"]) \
            .field("risk_count", fracture_risk["risk_count"]) \
            .field("risk_fraction", fracture_risk["risk_fraction"]) \
            .field("min_thickness_um", fracture_risk["min_thickness_um"]) \
            .time(now)
        points.append(p3)

        if action is not None:
            p4 = influxdb_client.Point("rl_optimization") \
                .tag("foil_id", self.foil_id).tag("session_id", self.session_id) \
                .field("rl_reward", reward).time(now)
            points.append(p4)

        self._write_points(points)

    def _report_minute_summary(self, path_name: str, force_name: str):
        current = time.time()
        elapsed = current - self.last_report_time
        if elapsed < REPORT_INTERVAL_SEC:
            return None
        metrics = self.physics.get_uniformity_metrics()
        summary = {
            "ts": datetime.now().isoformat(),
            "foil_id": self.foil_id,
            "session_id": self.session_id,
            "path": path_name, "force_profile": force_name,
            "strikes_period": self.strikes_in_current_minute,
            "total_strikes": self.physics.strike_count,
            "avg_thickness_um": metrics["mean_thickness_um"],
            "cv": metrics["coefficient_of_variation"],
            "uniformity_10pct": metrics["uniformity_within_10pct"],
            "elongation": self.physics.total_elongation,
            "grid_size": self.physics.grid_size,
        }
        print(
            f"\n  [{summary['ts']}] 锤击={summary['total_strikes']}  |"
            f" h={summary['avg_thickness_um']:.2f}μm  "
            f"CV={summary['cv']:.4f}  "
            f"延展={summary['elongation']:.2f}x  "
            f"网格={summary['grid_size']}"
        )
        self._write_thickness_snapshot()
        self.history_log.append(summary)
        self.strikes_in_current_minute = 0
        self.last_report_time = current
        return summary

    def _write_thickness_snapshot(self):
        if not self.use_influxdb:
            return
        h = self.physics.thickness_um
        step = max(1, h.shape[0] // 16)
        points = []
        now = datetime.now(timezone.utc)
        for i in range(0, h.shape[0], step):
            for j in range(0, h.shape[1], step):
                x = (j / h.shape[1] - 0.5) * FOIL_SIZE_MM
                y = (i / h.shape[0] - 0.5) * FOIL_SIZE_MM
                p = influxdb_client.Point("thickness_snapshot") \
                    .tag("foil_id", self.foil_id).tag("session_id", self.session_id) \
                    .field("x_mm", float(x)).field("y_mm", float(y)) \
                    .field("thickness_um", float(h[i, j])).time(now)
                points.append(p)
        self._write_points(points)

    def _apply_hammer_direct(
        self, force: float, pos: Tuple[float, float], radius: float
    ) -> Dict[str, Any]:
        hammer = HammerParameters(force=force, position=pos, radius_mm=radius)
        return self.physics.apply_hammer_strike(hammer)

    def run_with_profile(
        self,
        path_type: str = "center_out",
        force_profile: str = "medium",
        num_strikes: int = 200,
        strike_interval: float = 1.0,
        anneal_interval: int = 60,
        anneal_temp: float = 420.0,
        custom_sequence: Optional[List[StrikeEntry]] = None,
    ):
        self.running = True
        idx = 0

        if custom_sequence:
            sequence = custom_sequence
            num_strikes = len(sequence)
            print(f"[SIM] 使用自定义锤击序列: {num_strikes} 步")
        else:
            path_pts = generate_path_points(path_type, num_strikes)
            sequence = [
                StrikeEntry(
                    pos_x=p[0], pos_y=p[1],
                    force=None, radius_mm=None, wait_sec=strike_interval,
                    label=f"{path_type}/{i}"
                ) for i, p in enumerate(path_pts)
            ]
            print(f"[SIM] 路径: {path_type} ({PATH_PRESETS.get(path_type,'')})")
            print(f"[SIM] 力度: {force_profile} ({FORCE_PRESETS[force_profile]['desc']})")

        force_cfg = FORCE_PRESETS.get(force_profile, FORCE_PRESETS["medium"])

        print(f"[SIM] 金箔ID={self.foil_id} | 工匠={self.craftsman_id}")
        print(f"[SIM] 网格={self.physics.grid_size} | 尺寸={FOIL_SIZE_MM}mm | 初始厚度={self.physics.thickness_um.mean():.1f}μm")
        print(f"[SIM] 锤击={num_strikes}次 | 退火每{anneal_interval}次 | 温度={anneal_temp}°C")
        print(f"[SIM] InfluxDB={'启用' if self.use_influxdb else '禁用'} | API={self.api_base or '无'}")
        print("-" * 60)

        try:
            while self.running and idx < num_strikes:
                entry = sequence[idx]

                if idx > 0 and idx % anneal_interval == 0:
                    avg_strain = float(self.physics.plastic_strain.mean())
                    if avg_strain > force_cfg["anneal_strain"]:
                        print(f"[ANNEAL] #{idx} 应变={avg_strain:.3f}, 退火{anneal_temp}°C...")
                        self.physics.apply_annealing(anneal_temp, duration_min=5.0)

                if path_type == "heuristic":
                    action, strike, reward, risk = self.rl_session.step(ActionType.HEURISTIC)
                    self._persist_step(strike, risk, action, reward, path_label="heuristic")
                elif path_type == "rl":
                    action, strike, reward, risk = self.rl_session.step(ActionType.Q_LEARNING)
                    self._persist_step(strike, risk, action, reward, path_label="rl")
                elif path_type == "pretrained":
                    action, strike, reward, risk = self.rl_session.step(ActionType.PRETRAINED)
                    self._persist_step(strike, risk, action, reward, path_label="pretrained")
                else:
                    force = entry.force
                    radius = entry.radius_mm or force_cfg["radius_mm"]
                    if force is None:
                        force, _ = generate_force_value(
                            force_profile, idx, num_strikes, self.physics
                        )
                    pos = (entry.pos_x, entry.pos_y)
                    strike = self._apply_hammer_direct(force, pos, radius)
                    risk = self.physics.check_fracture_risk(0.1)
                    self._persist_step(strike, risk, path_label=entry.label or path_type)

                self.strikes_in_current_minute += 1
                idx += 1

                self._report_minute_summary(path_type, force_profile)

                if risk["risk_level"] == "high":
                    print(f"[ALERT] #{idx} 破裂风险HIGH! 最薄={risk['min_thickness_um']:.4f}μm")

                if strike["avg_thickness_um"] < 0.15:
                    print(f"\n[DONE] 达到目标厚度: {strike['avg_thickness_um']:.4f}μm (共{idx}次锤击)")
                    break

                time.sleep(max(0, entry.wait_sec if custom_sequence else strike_interval))

        except KeyboardInterrupt:
            print("\n[SIM] 用户中断")
        finally:
            self.running = False

        return self._finalize(idx)

    def _finalize(self, total: int) -> Dict[str, Any]:
        m = self.physics.get_uniformity_metrics()
        print("\n" + "=" * 60)
        print(f"[SIM] 结束 | 总锤击={total}")
        print(f"  厚度: {m['mean_thickness_um']:.3f} ± {m['std_thickness_um']:.3f} μm")
        print(f"  CV:     {m['coefficient_of_variation']:.4f}")
        print(f"  ±10%:   {m['uniformity_within_10pct']*100:.1f}%")
        print(f"  延展:   {self.physics.total_elongation:.2f}x")
        print(f"  网格:   {self.physics.grid_size}x{self.physics.grid_size}")
        print("=" * 60)
        self.close()
        return {"strikes": total, "final_metrics": m}

    def close(self):
        if self.influx_client:
            self.influx_client.close()


def list_presets():
    print("\n" + "=" * 60)
    print("可用工艺预设 (--preset <name>):")
    print("=" * 60)
    for name, p in PRESET_PROFILES.items():
        print(f"  {name:18s}  {p['description']}")
    print("\n可用路径预设 (--path <name>):")
    for name, desc in PATH_PRESETS.items():
        print(f"  {name:18s}  {desc}")
    print("\n可用力度预设 (--force <name>):")
    for name, cfg in FORCE_PRESETS.items():
        print(f"  {name:18s}  {cfg['desc']} ({cfg['min']:.0f}-{cfg['max']:.0f}N)")


def main():
    import argparse
    p = argparse.ArgumentParser(description="金箔锻制工艺模拟器 (v3 工程化版)")
    p.add_argument("--preset", type=str, default=None, help="工艺预设名称, --list-presets查看")
    p.add_argument("--list-presets", action="store_true", help="列出所有预设并退出")
    p.add_argument("--path", type=str, default="center_out", help="锤击路径类型")
    p.add_argument("--force", type=str, default="medium", help="力度预设")
    p.add_argument("--strikes", type=int, default=None, help="总锤击次数")
    p.add_argument("--interval", type=float, default=1.0, help="锤击间隔(秒)")
    p.add_argument("--anneal-every", type=int, default=None, help="每N次锤击触发退火")
    p.add_argument("--anneal-temp", type=float, default=420, help="退火温度(°C)")
    p.add_argument("--custom-path", type=str, default=None, help="自定义JSON锤击序列路径")
    p.add_argument("--foil-id", type=str, default=FOIL_ID)
    p.add_argument("--craftsman", type=str, default=CRAFTSMAN_ID)
    p.add_argument("--grid-size", type=int, default=48)
    p.add_argument("--initial-thickness", type=float, default=None, help="初始厚度μm")
    p.add_argument("--no-influxdb", action="store_true")
    args = p.parse_args()

    if args.list_presets:
        list_presets()
        return

    profile = PRESET_PROFILES.get(args.preset) if args.preset else None
    if profile:
        path_type = profile["path"]
        force_p = profile["force"]
        strikes = args.strikes or profile["strikes"]
        interval = args.interval
        anneal_int = args.anneal_every or profile["anneal_interval_strikes"]
        anneal_t = args.anneal_temp if args.anneal_temp != 420 else profile["anneal_temp"]
        init_t = args.initial_thickness or profile["initial_thickness"]
    else:
        path_type = args.path
        force_p = args.force
        strikes = args.strikes or 200
        interval = args.interval
        anneal_int = args.anneal_every or 60
        anneal_t = args.anneal_temp
        init_t = args.initial_thickness

    custom_seq = None
    if args.custom_path:
        custom_seq = load_custom_path(args.custom_path)
        strikes = len(custom_seq)

    sim = GoldFoilSimulator(
        foil_id=args.foil_id,
        craftsman_id=args.craftsman,
        grid_size=args.grid_size,
        use_influxdb=not args.no_influxdb,
        api_base=API_BASE,
    )

    if init_t is not None:
        sim.physics.thickness_um = np_full_init(
            sim.physics.grid_size, init_t
        )
        print(f"[SIM] 设置初始厚度: {init_t:.1f}μm")

    sim.run_with_profile(
        path_type=path_type,
        force_profile=force_p,
        num_strikes=strikes,
        strike_interval=interval,
        anneal_interval=anneal_int,
        anneal_temp=anneal_t,
        custom_sequence=custom_seq,
    )


def np_full_init(size, value):
    import numpy as np
    return np.full((size, size), value, dtype=np.float64)


if __name__ == "__main__":
    main()
