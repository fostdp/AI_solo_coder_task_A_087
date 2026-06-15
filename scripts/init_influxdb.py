"""
InfluxDB 初始化脚本 (v3 工程化版)
1. 创建 Org、Bucket、Token
2. 创建 4 个 downsampling 任务:
   - raw       -> 1min  聚合 (保留 7 天)
   - 1min_agg  -> 1hour 聚合 (保留 30 天)
   - 1hour_agg -> 1day  聚合 (保留 365 天)
3. 创建 Grafana 可用的 DBRP 映射

运行: python scripts/init_influxdb.py
      或 docker 启动时挂载自动执行
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import influxdb_client
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client import (
    InfluxDBClient, Organization, Bucket, BucketRetentionRules,
    Permission, Resource, Authorization, DBRPs, DBRP,
)
from influxdb_client.rest import ApiException


INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_USER = os.getenv("INFLUXDB_USER", "admin")
INFLUXDB_PASS = os.getenv("INFLUXDB_PASS", "gold-foil-admin-pass")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "craftsman-research")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "gold-foil-data")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "gold-foil-simulation-token")

BUCKETS_SPEC = [
    {"name": f"{INFLUXDB_BUCKET}_raw",     "retention_seconds": 7 * 24 * 3600,       "desc": "原始锤击数据 (保留7天)"},
    {"name": f"{INFLUXDB_BUCKET}_1m",      "retention_seconds": 30 * 24 * 3600,      "desc": "1分钟聚合数据 (保留30天)"},
    {"name": f"{INFLUXDB_BUCKET}_1h",      "retention_seconds": 365 * 24 * 3600,     "desc": "1小时聚合数据 (保留365天)"},
    {"name": f"{INFLUXDB_BUCKET}_1d",      "retention_seconds": 5 * 365 * 24 * 3600, "desc": "1天聚合数据 (保留5年)"},
    {"name": INFLUXDB_BUCKET,              "retention_seconds": 0,                   "desc": "主bucket (无期限，供读写通用)"},
]

MEASUREMENTS = ["forging_metrics", "uniformity_metrics", "fracture_risk", "rl_optimization", "thickness_snapshot"]


def _wait_influxdb(client: InfluxDBClient, max_attempts: int = 30) -> bool:
    for i in range(1, max_attempts + 1):
        try:
            health = client.health()
            if health.status == "pass":
                print(f"[OK] InfluxDB ready (attempt {i})")
                return True
            print(f"[WAIT] InfluxDB not ready, status={health.status}, attempt {i}/{max_attempts}")
        except Exception as e:
            print(f"[WAIT] InfluxDB connection attempt {i}/{max_attempts}: {type(e).__name__}")
        time.sleep(3)
    return False


def _ensure_setup(client: InfluxDBClient) -> tuple[str, str]:
    """使用 onboarding API 完成首次设置，返回 (org_id, token)"""
    setup_api = client.setup_api()
    if setup_api.is_onboarding_allowed():
        print("[INIT] 首次启动，执行 InfluxDB onboarding...")
        result = setup_api.setup(
            username=INFLUXDB_USER,
            password=INFLUXDB_PASS,
            org=INFLUXDB_ORG,
            bucket=INFLUXDB_BUCKET,
            token=INFLUXDB_TOKEN,
        )
        org_id = result.org.id
        token = result.auth.token
        print(f"[OK] Onboarded org={INFLUXDB_ORG} id={org_id}")
    else:
        print("[INIT] InfluxDB 已配置，查找 org 和 operator token...")
        orgs_api = client.organizations_api()
        orgs = orgs_api.find_organizations(org=INFLUXDB_ORG)
        if not orgs:
            raise RuntimeError(f"Organization {INFLUXDB_ORG} 未找到")
        org_id = orgs[0].id
        auths_api = client.authorizations_api()
        token = INFLUXDB_TOKEN
        matching = [a for a in auths_api.find_authorizations(org_id=org_id) if a.token == token]
        if matching:
            print(f"[OK] 使用现有 token (org={INFLUXDB_ORG})")
        else:
            print("[INFO] Token 不存在，创建新的 all-access token...")
            auth = Authorization(org_id=org_id, description="gold-foil-all-access", permissions=[
                Permission(resource=Resource(type="orgs", id=org_id, orgID=org_id), action="write"),
                Permission(resource=Resource(type="orgs", id=org_id, orgID=org_id), action="read"),
                Permission(resource=Resource(type="buckets", orgID=org_id), action="write"),
                Permission(resource=Resource(type="buckets", orgID=org_id), action="read"),
                Permission(resource=Resource(type="tasks", orgID=org_id), action="write"),
                Permission(resource=Resource(type="tasks", orgID=org_id), action="read"),
                Permission(resource=Resource(type="telegrafs", orgID=org_id), action="write"),
                Permission(resource=Resource(type="telegrafs", orgID=org_id), action="read"),
                Permission(resource=Resource(type="dashboards", orgID=org_id), action="write"),
                Permission(resource=Resource(type="dashboards", orgID=org_id), action="read"),
                Permission(resource=Resource(type="dbrp", orgID=org_id), action="write"),
                Permission(resource=Resource(type="dbrp", orgID=org_id), action="read"),
                Permission(resource=Resource(type="scrapers", orgID=org_id), action="write"),
                Permission(resource=Resource(type="scrapers", orgID=org_id), action="read"),
                Permission(resource=Resource(type="secrets", orgID=org_id), action="write"),
                Permission(resource=Resource(type="secrets", orgID=org_id), action="read"),
                Permission(resource=Resource(type="labels", orgID=org_id), action="write"),
                Permission(resource=Resource(type="labels", orgID=org_id), action="read"),
                Permission(resource=Resource(type="variables", orgID=org_id), action="write"),
                Permission(resource=Resource(type="variables", orgID=org_id), action="read"),
            ])
            created = auths_api.create_authorization(auth)
            token = created.token
            with open(os.environ.get("INFLUXDB_TOKEN_FILE", "/tmp/influx_token.txt"), "w") as f:
                f.write(token)
            print(f"[OK] 新 token 已生成并保存")
    return org_id, token


def _ensure_buckets(client: InfluxDBClient, org_id: str) -> dict:
    buckets_api = client.buckets_api()
    existing = {b.name: b for b in buckets_api.find_buckets(org=INFLUXDB_ORG).buckets}
    created = {}
    for spec in BUCKETS_SPEC:
        if spec["name"] in existing:
            created[spec["name"]] = existing[spec["name"]]
            print(f"[SKIP] Bucket exists: {spec['name']}")
            continue
        retention = None
        if spec["retention_seconds"] > 0:
            retention = BucketRetentionRules(type="expire", every_seconds=spec["retention_seconds"])
        b = Bucket(
            name=spec["name"],
            org_id=org_id,
            description=spec["desc"],
            retention_rules=[retention] if retention else [],
        )
        created[spec["name"]] = buckets_api.create_bucket(b)
        print(f"[OK] 创建 bucket: {spec['name']} ({spec['desc']})")
    return created


def _build_agg_flux(src_bucket: str, dst_bucket: str, every: str, offset: str = "0s") -> str:
    agg_field_block = []
    for m in MEASUREMENTS:
        block = f"""
  |> filter(fn: (r) => r._measurement == "{m}")
  |> filter(fn: (r) => r._field != "grid_size")
  |> aggregateWindow(every: {every}, offset: {offset}, fn: mean, createEmpty: false)
  |> to(bucket: "{dst_bucket}", org: "{INFLUXDB_ORG}")"""
        agg_field_block.append(block)
    joined = "\n".join(agg_field_block)
    return f"""option task = {{name: "downsample_{every}", every: {every}, offset: {offset}}}

data = from(bucket: "{src_bucket}")
  |> range(start: -task.every)
{joined}
"""


def _ensure_task(client: InfluxDBClient, org_id: str, name: str, flux: str) -> None:
    tasks_api = client.tasks_api()
    existing = [t for t in tasks_api.find_tasks(org=INFLUXDB_ORG) if t.name == name]
    if existing:
        print(f"[SKIP] Task exists: {name}")
        return
    try:
        tasks_api.create_task(flux=flux, description=name, org_id=org_id)
        print(f"[OK] 创建任务: {name}")
    except ApiException as e:
        print(f"[WARN] 创建任务 {name} 失败（可忽略）: {e.status}")


def _ensure_dbrp(client: InfluxDBClient, org_id: str, buckets: dict) -> None:
    dbrp_api = client.dbrp_service()
    existing = set()
    try:
        list_resp = dbrp_api.get_dbr_ps(org_id=org_id)
        for item in getattr(list_resp, "content", []) or []:
            existing.add((item.database, item.retention_policy))
    except Exception:
        pass
    for b_name, b in buckets.items():
        db_name = b_name
        rp_name = "autogen"
        if (db_name, rp_name) in existing:
            continue
        try:
            dbrp_api.post_dbr_p(DBRP(
                org_id=org_id,
                bucket_id=b.id,
                database=db_name,
                retention_policy=rp_name,
                default=True,
            ))
            print(f"[OK] DBRP映射: {db_name}.{rp_name} -> {b_name}")
        except Exception as e:
            print(f"[WARN] DBRP {db_name}.{rp_name} 创建失败: {e}")


def main():
    print("=" * 60)
    print(f"InfluxDB 初始化 - {INFLUXDB_URL}")
    print("=" * 60)

    admin_client = InfluxDBClient(url=INFLUXDB_URL, token="", org=INFLUXDB_ORG, debug=False)
    if not _wait_influxdb(admin_client):
        print("[FATAL] InfluxDB 未就绪，退出")
        sys.exit(1)

    org_id, token = _ensure_setup(admin_client)

    client = InfluxDBClient(url=INFLUXDB_URL, token=token, org=INFLUXDB_ORG, debug=False)
    buckets = _ensure_buckets(client, org_id)

    raw_b = buckets[INFLUXDB_BUCKET].name if INFLUXDB_BUCKET in buckets else f"{INFLUXDB_BUCKET}_raw"
    b_1m = f"{INFLUXDB_BUCKET}_1m"
    b_1h = f"{INFLUXDB_BUCKET}_1h"
    b_1d = f"{INFLUXDB_BUCKET}_1d"

    _ensure_task(client, org_id, "downsample_raw_to_1min",
                 _build_agg_flux(raw_b, b_1m, every="1m", offset="30s"))
    _ensure_task(client, org_id, "downsample_1min_to_1hour",
                 _build_agg_flux(b_1m, b_1h, every="1h", offset="5m"))
    _ensure_task(client, org_id, "downsample_1hour_to_1day",
                 _build_agg_flux(b_1h, b_1d, every="1d", offset="15m"))

    _ensure_dbrp(client, org_id, buckets)

    print("=" * 60)
    print(f"[DONE] InfluxDB 初始化完成")
    print(f"  Org:   {INFLUXDB_ORG} ({org_id})")
    print(f"  Token: {token[:12]}..." + (" (matches env)" if token == INFLUXDB_TOKEN else " (使用生成的token)"))
    print(f"  主 Bucket: {INFLUXDB_BUCKET}")
    print(f"  Downsampling: raw->1m->{b_1h}->{b_1d}")
    print("=" * 60)
    client.close()
    admin_client.close()


if __name__ == "__main__":
    main()
