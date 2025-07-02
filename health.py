import psutil
import time
import requests
import json

def snapshot() -> dict:
    """
    Take a snapshot of system health stats (CPU, memory, disk).
    """
    return {
        "ts": time.time(),
        "cpu": psutil.cpu_percent(),
        "mem": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
    }

def latency(url: str = "https://duckduckgo.com") -> float:
    """
    Measure HTTP latency to a URL in milliseconds.
    Returns -1 if unreachable.
    """
    try:
        t0 = time.time()
        requests.get(url, timeout=5)
        return (time.time() - t0) * 1000
    except requests.RequestException:
        return -1

def run() -> dict:
    """
    Print and return system health info plus latency.
    """
    data = snapshot()
    data["lat_ms"] = latency()
    print(json.dumps(data, indent=2))
    return data

if __name__ == "__main__":
    run()
