import os
import json
import requests

LOKI_URL = "http://localhost:3100/loki/api/v1/query_range"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "log")

def fetch_loki_logs(pod_name, duration="1h"):
    """Fetch logs from Loki."""
    query = f'{{pod="{pod_name}"}}'
    try:
        res = requests.get(LOKI_URL, params={"query": query, "limit": 1000}, timeout=10)
        res.raise_for_status()
        results = []
        for stream in res.json().get("data", {}).get("result", []):
            results.extend(stream.get("values", []))
        return results
    except Exception as e:
        print(f"Loki Error ({pod_name}): {e}")
        return []

def abstract_logs(raw_logs):
    """Filter and deduplicate error logs."""
    errors = []
    seen = set()
    for entry in raw_logs:
        # Loki logs are [timestamp, message]
        msg = entry[1] if isinstance(entry, list) and len(entry) > 1 else str(entry)
        if any(w in msg.upper() for w in ["ERROR", "EXCEPTION", "FAIL", "CRITICAL"]):
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)
    return errors

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    for pod in ["app-a", "app-b", "app-c"]:
        logs = fetch_loki_logs(pod)
        if not logs:
            sample_path = os.path.join(LOG_DIR, "sample_raw_logs.json")
            if os.path.exists(sample_path):
                with open(sample_path, "r") as f:
                    sample_data = json.load(f)
                    # Sample data is a list of streams
                    logs = []
                    for stream in sample_data:
                        if stream.get("stream", {}).get("compose_service") == pod:
                            logs.extend(stream.get("values", []))
        
        abstracted = abstract_logs(logs)
        out_path = os.path.join(LOG_DIR, f"abstracted_logs_{pod}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(abstracted, f, indent=2)
        print(f"[OK] Logs saved: {out_path}")

if __name__ == "__main__":
    main()
