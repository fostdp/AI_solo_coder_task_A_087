"""
Gunicorn 配置文件 - 生产环境部署
配合 UvicornWorker 运行 FastAPI 异步应用
"""
import multiprocessing
import os

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

workers_per_core = int(os.getenv("GUNICORN_WORKERS_PER_CORE", "2"))
cores = multiprocessing.cpu_count()
default_workers = max(2, min(cores * workers_per_core, 8))

workers = int(os.getenv("GUNICORN_WORKERS", str(default_workers)))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")

worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "20"))

backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)ss'

preload_app = os.getenv("GUNICORN_PRELOAD", "false").lower() == "true"

max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "10000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "1000"))

reload = os.getenv("GUNICORN_RELOAD", "false").lower() == "true"

def when_ready(server):
    server.log.info(
        f"Gunicorn ready: workers={workers}, "
        f"worker_class={worker_class}, bind={bind}"
    )

def worker_int(worker):
    worker.log.info(f"worker pid={worker.pid} received INT signal, exiting")

def post_fork(server, worker):
    import numpy as np
    np.random.seed(worker.pid % 2**32)
