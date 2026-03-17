import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

PROM_URL = "http://localhost:9090"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "log", "metric_profiling")
METRIC_QUERIES = {
    "cpu_proxy": 'rate(duration_milliseconds_sum[1m]) / rate(duration_milliseconds_count[1m])',
    "io_proxy": 'rate(calls_total[1m])',
}

def query_prometheus(query, mins=30):
    """Fetch range data from Prometheus."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=mins)
    params = {"query": query, "start": start.isoformat(), "end": now.isoformat(), "step": "15s"}
    try:
        res = requests.get(f"{PROM_URL}/api/v1/query_range", params=params, timeout=10)
        res.raise_for_status()
        return res.json()["data"]["result"]
    except Exception as e:
        print(f"Prometheus Error: {e}")
    return None

def process_metrics(raw):
    """Summarize raw metrics into 2-minute stats."""
    if not raw: return {}
    pod_data = {}
    for res in raw:
        pod = next((res["metric"][k] for k in ["pod", "service_name", "job"] if k in res["metric"]), "unknown")
        df = pd.DataFrame(res["values"], columns=["ts", "val"])
        df["ts"] = pd.to_datetime(df["ts"], unit='s')
        df["val"] = df["val"].astype(float)
        df = df.set_index("ts")
        resampled = df["val"].resample('2min')
        pod_data[pod] = pd.DataFrame({"mean": resampled.mean(), "max": resampled.max()})
    return pod_data

def generate_charts(all_data):
    """Save consolidated PNG charts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, data in all_data.items():
        if not data: continue
        plt.figure(figsize=(10, 6))
        for pod, df in data.items():
            plt.plot(df.index, df["mean"], label=f"{pod}", marker='o')
        plt.title(f"Metric: {name.upper()}")
        plt.legend()
        plt.grid(True)
        path = os.path.join(OUTPUT_DIR, f"consolidated_{name}.png")
        plt.savefig(path)
        plt.close()
        print(f"[OK] Chart: {path}")

def main():
    all_data = {n: process_metrics(query_prometheus(q)) for n, q in METRIC_QUERIES.items()}
    generate_charts(all_data)

if __name__ == "__main__":
    main()
