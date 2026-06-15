"""
共享工具模块：配置加载 & Redis消息总线
"""
import os
import json
from typing import Optional, Dict, Any
from dataclasses import asdict

import redis


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

REDIS_CHANNELS = {
    "sensor_raw": "sensor:raw",
    "hammer_request": "hammer:request",
    "strike_result": "strike:result",
    "thickness_updated": "thickness:updated",
    "rl_action_request": "rl:action:request",
    "rl_action_result": "rl:action:result",
    "alarm_triggered": "alarm:triggered",
    "system_event": "system:event",
    "mesh_quality": "mesh:quality",
}


def load_json_config(filename: str) -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_material_config() -> Dict[str, Any]:
    return load_json_config("material.json")


def load_rl_config() -> Dict[str, Any]:
    return load_json_config("rl_config.json")


class RedisBus:
    """
    Redis Pub/Sub 消息总线
    - 可作为发布者使用 publish()
    - 可作为订阅者使用 subscribe() + listen()
    - Redis不可用时降级为内存模式（方便开发/测试）
    """

    def __init__(self, url: Optional[str] = None):
        self.url = url or REDIS_URL
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._in_memory_handlers: Dict[str, list] = {}
        self.available: bool = False
        self._connect()

    def _connect(self):
        try:
            self._client = redis.from_url(self.url, decode_responses=True)
            self._client.ping()
            self._pubsub = self._client.pubsub()
            self.available = True
        except Exception as e:
            print(f"[WARN] Redis连接失败，降级为内存模式: {e}")
            self.available = False
            self._client = None
            self._pubsub = None

    def publish(self, channel: str, payload: Dict[str, Any]):
        msg = json.dumps(payload, ensure_ascii=False, default=str)
        if self.available and self._client:
            try:
                self._client.publish(channel, msg)
            except Exception:
                self._fallback_publish(channel, payload)
        else:
            self._fallback_publish(channel, payload)

    def _fallback_publish(self, channel: str, payload: Dict[str, Any]):
        handlers = self._in_memory_handlers.get(channel, [])
        for handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                print(f"[ERROR] 内存消息处理器异常 [{channel}]: {e}")

    def subscribe(self, channel: str, handler):
        if self.available and self._pubsub:
            try:
                self._pubsub.subscribe(**{channel: lambda msg: self._redis_handler(msg, handler)})
            except Exception:
                self._in_memory_handlers.setdefault(channel, []).append(handler)
        else:
            self._in_memory_handlers.setdefault(channel, []).append(handler)

    def _redis_handler(self, msg, handler):
        if msg["type"] != "message":
            return
        try:
            data = json.loads(msg["data"])
            handler(data)
        except Exception as e:
            print(f"[ERROR] Redis消息解析失败: {e}")

    def start_listen_thread(self):
        if self.available and self._pubsub:
            import threading
            t = threading.Thread(target=self._listen_loop, daemon=True)
            t.start()
            return t
        return None

    def _listen_loop(self):
        try:
            for item in self._pubsub.listen():
                pass
        except Exception as e:
            print(f"[ERROR] Redis监听线程异常: {e}")

    def get_client(self) -> Optional[redis.Redis]:
        return self._client

    def close(self):
        if self._pubsub:
            try:
                self._pubsub.close()
            except Exception:
                pass
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
