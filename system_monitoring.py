# system_monitoring.py
import psutil
from elasticsearch import Elasticsearch
from datetime import datetime

es = Elasticsearch("http://localhost:9200")

def monitor_system():
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "docker_containers": len(psutil.process_iter(attrs=['name']))
    }
    
    es.index(index="system-metrics", body=metrics)
    print(f"System metrics logged: {metrics}")
