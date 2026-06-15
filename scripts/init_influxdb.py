"""
InfluxDB 初始化脚本
用于创建金箔锻制工艺仿真系统所需的Bucket和保留策略
"""
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import time

INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "gold-foil-simulation-token"
INFLUXDB_ORG = "craftsman-research"
INFLUXDB_BUCKET = "gold-foil-data"


def init_influxdb():
    """初始化InfluxDB连接，创建Bucket和初始数据"""
    print("[INFO] 正在连接 InfluxDB...")
    
    client = influxdb_client.InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG
    )
    
    health = client.health()
    print(f"[INFO] InfluxDB 状态: {health.status}")
    
    buckets_api = client.buckets_api()
    
    existing_bucket = None
    try:
        existing_bucket = buckets_api.find_bucket_by_name(INFLUXDB_BUCKET)
    except Exception:
        pass
    
    if existing_bucket:
        print(f"[INFO] Bucket '{INFLUXDB_BUCKET}' 已存在，跳过创建")
    else:
        print(f"[INFO] 创建 Bucket: {INFLUXDB_BUCKET}")
        buckets_api.create_bucket(
            bucket_name=INFLUXDB_BUCKET,
            description="南京金箔锻制工艺仿真数据 - 锤击力度、温度、厚度分布、延展率",
            org=INFLUXDB_ORG
        )
        print("[SUCCESS] Bucket 创建完成")
    
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()
    
    print("[INFO] 写入测试样例数据...")
    from datetime import datetime, timezone
    
    test_point = influxdb_client.Point("forging_metrics") \
        .tag("foil_id", "NF-001") \
        .tag("craftsman", "master_wu") \
        .field("hammer_force", 850.0) \
        .field("temperature", 450.0) \
        .field("avg_thickness", 15.5) \
        .field("thickness_std", 2.3) \
        .field("elongation_rate", 1.85) \
        .time(datetime.now(timezone.utc))
    
    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=test_point)
    print("[SUCCESS] 测试数据写入完成")
    
    print("\n" + "="*50)
    print("InfluxDB 初始化完成!")
    print(f"  URL: {INFLUXDB_URL}")
    print(f"  Org: {INFLUXDB_ORG}")
    print(f"  Bucket: {INFLUXDB_BUCKET}")
    print("="*50)
    
    client.close()


if __name__ == "__main__":
    init_influxdb()
