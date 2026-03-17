"""
GALA TWIST — Trace-Based Root Cause Ranking (Phase 1)
=====================================================
Fetch traces from Jaeger → build trace DAGs → rank services using TWIST:
  c1: Self-Anomaly Score  (ratio of anomalous spans)
  c2: Trace Impact Score  (ratio of error traces containing service)
  c3: Blast Radius Score  (downstream impact in DAG)
  c4: Delay Severity Score (normalized latency deviation)

  score(s) = Σ wi * ci(s)
"""

import os, json
from collections import defaultdict
import statistics
import requests
import networkx as nx

# ── CONFIG ───────────────────────────────────
JAEGER_URL = "http://localhost:16686"
SERVICE_NAME = "app-a"
LOOKBACK = "1h"
LIMIT = 20
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "log")

# TWIST weights
W1, W2, W3, W4 = 0.3, 0.25, 0.25, 0.2

# Anomaly: span duration > mean + Z_THR * std
Z_THR = 2.0


# ── 1. FETCH TRACES ─────────────────────────
def fetch_traces(service=SERVICE_NAME, lookback=LOOKBACK, limit=LIMIT):
    """Fetch traces from Jaeger API."""
    r = requests.get(f"{JAEGER_URL}/api/traces", params={
        "service": service, "lookback": lookback, "limit": limit
    })
    r.raise_for_status()
    traces = r.json().get("data", [])
    # save raw
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "traces.json"), "w", encoding="utf-8") as f:
        json.dump({"data": traces}, f, ensure_ascii=False, indent=2)
    print(f"📡 Fetched {len(traces)} traces from Jaeger (service={service})")
    return traces


# ── 2. BUILD TRACE DAG ──────────────────────
def build_trace_dag(trace):
    """Build service-level DAG from a single trace. Returns (G, span_info)."""
    G = nx.DiGraph()
    spans = trace["spans"]
    procs = trace["processes"]
    span_map = {}

    for span in spans:
        sid = span["spanID"]
        svc = procs[span["processID"]]["serviceName"]
        dur = span["duration"] / 1000.0  # ms
        tags = {t["key"]: t["value"] for t in span.get("tags", [])}
        has_error = tags.get("error", False) or (isinstance(tags.get("http.status_code", 200), int) and tags.get("http.status_code", 200) >= 500)

        span_map[sid] = {"service": svc, "duration": dur, "error": has_error}

        if svc not in G:
            G.add_node(svc, durations=[], errors=0, total_spans=0)
        G.nodes[svc]["durations"].append(dur)
        G.nodes[svc]["total_spans"] += 1
        if has_error:
            G.nodes[svc]["errors"] += 1

    # edges: parent → child (different services only)
    for span in spans:
        child_svc = span_map[span["spanID"]]["service"]
        for ref in span.get("references", []):
            if ref["refType"] == "CHILD_OF" and ref["spanID"] in span_map:
                parent_svc = span_map[ref["spanID"]]["service"]
                if parent_svc != child_svc:
                    G.add_edge(parent_svc, child_svc)

    return G, span_map


# ── 3. TWIST SCORING ────────────────────────
def twist_score(traces):
    """
    Compute TWIST scores for each service across all traces.
    Returns: [(service, {score, c1, c2, c3, c4, ...})] sorted desc.
    """
    # Aggregate stats across traces
    svc_stats = defaultdict(lambda: {
        "durations": [], "anomalous_spans": 0, "total_spans": 0,
        "error_traces": 0, "total_traces": 0, "child_counts": [],
    })
    error_trace_count = 0

    for trace in traces:
        G, span_map = build_trace_dag(trace)

        # Check if trace has any error
        trace_has_error = any(s["error"] for s in span_map.values())
        if trace_has_error:
            error_trace_count += 1

        services_in_trace = set()
        for sid, info in span_map.items():
            svc = info["service"]
            services_in_trace.add(svc)
            svc_stats[svc]["durations"].append(info["duration"])
            svc_stats[svc]["total_spans"] += 1

        # Track which services appear in error traces
        if trace_has_error:
            for svc in services_in_trace:
                svc_stats[svc]["error_traces"] += 1

        svc_stats_keys = set(svc_stats.keys())
        for svc in services_in_trace:
            svc_stats[svc]["total_traces"] += 1
            # blast radius: count children in DAG
            if svc in G:
                svc_stats[svc]["child_counts"].append(len(list(nx.descendants(G, svc))))

    # Learn dynamic thresholds per service (mean + Z_THR * std)
    for svc, st in svc_stats.items():
        durs = st["durations"]
        if len(durs) > 1:
            mu, sigma = statistics.mean(durs), statistics.stdev(durs)
            threshold = mu + Z_THR * sigma
            st["anomalous_spans"] = sum(1 for d in durs if d > threshold)
        else:
            st["anomalous_spans"] = 0

    # ── Compute c1-c4 (all normalized to [0, 1]) ──

    # For c4 normalization: max deviation across all services
    max_dev = 0
    deviations = {}
    for svc, st in svc_stats.items():
        durs = st["durations"]
        if len(durs) > 1:
            mu = statistics.mean(durs)
            dev = max(abs(d - mu) for d in durs)
        else:
            dev = 0
        deviations[svc] = dev
        max_dev = max(max_dev, dev)

    # For c3 normalization: max blast radius
    max_blast = max((max(st["child_counts"]) if st["child_counts"] else 0) for st in svc_stats.values()) or 1

    results = []
    for svc, st in svc_stats.items():
        # c1: Self-Anomaly Score
        c1 = st["anomalous_spans"] / st["total_spans"] if st["total_spans"] > 0 else 0

        # c2: Trace Impact Score
        c2 = st["error_traces"] / error_trace_count if error_trace_count > 0 else 0

        # c3: Blast Radius Score
        avg_blast = statistics.mean(st["child_counts"]) if st["child_counts"] else 0
        c3 = avg_blast / max_blast if max_blast > 0 else 0

        # c4: Delay Severity Score
        c4 = deviations[svc] / max_dev if max_dev > 0 else 0

        score = W1 * c1 + W2 * c2 + W3 * c3 + W4 * c4

        avg_dur = statistics.mean(st["durations"]) if st["durations"] else 0

        results.append((svc, {
            "score": round(score, 6),
            "c1_self_anomaly": round(c1, 4),
            "c2_trace_impact": round(c2, 4),
            "c3_blast_radius": round(c3, 4),
            "c4_delay_severity": round(c4, 4),
            "avg_duration_ms": round(avg_dur, 2),
            "total_spans": st["total_spans"],
            "anomalous_spans": st["anomalous_spans"],
            "error_traces": st["error_traces"],
        }))

    results.sort(key=lambda x: x[1]["score"], reverse=True)
    return results


# ── 4. PIPELINE ─────────────────────────────
def run_pipeline(service=SERVICE_NAME, lookback=LOOKBACK, limit=LIMIT):
    print("\n" + "█" * 50)
    print("  GALA TWIST — TRACE RCA PIPELINE")
    print("█" * 50)

    traces = fetch_traces(service, lookback, limit)
    if not traces:
        print("❌ No traces found!")
        return {"error": "No traces"}

    ranking = twist_score(traces)

    # Print ranking
    print(f"\n🏆 TWIST Ranking (w={W1},{W2},{W3},{W4}):")
    print(f"  {'#':<3} {'Service':<20} {'Score':<8} {'c1(self)':<10} {'c2(impact)':<10} {'c3(blast)':<10} {'c4(delay)':<10}")
    print(f"  {'─'*3} {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for i, (svc, v) in enumerate(ranking, 1):
        rc = " ◀ ROOT CAUSE" if i == 1 else ""
        print(f"  {i:<3} {svc:<20} {v['score']:<8.4f} {v['c1_self_anomaly']:<10.4f} "
              f"{v['c2_trace_impact']:<10.4f} {v['c3_blast_radius']:<10.4f} {v['c4_delay_severity']:<10.4f}{rc}")

    # Save JSON
    os.makedirs(LOG_DIR, exist_ok=True)
    out = {
        "ranking": [{"rank": i+1, "service": svc, **v} for i, (svc, v) in enumerate(ranking)],
    }
    path = os.path.join(LOG_DIR, "trace_rca_ranking.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n Done → {path}")
    if ranking:
        rc, info = ranking[0]
        print(f"🔥 Root cause: {rc} (score={info['score']:.4f})")
    return {"ranking": ranking}


if __name__ == "__main__":
    run_pipeline()