"""
GALA-Style Metric-Based Causal RCA — Phase 1
=============================================
1. Fetch metrics from Prometheus (request rate, error rate, latency)
2. Z-score anomaly detection
3. Build causal DAG via lagged cross-correlation
4. Rank root causes via Personalized PageRank
5. Save results to rca/log/
"""

import os, json
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats as sp_stats

# ── CONFIG ───────────────────────────────────
PROM = "http://localhost:9090"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "log")

Z_THRESHOLD = 2.0
ROLLING_WINDOW = 10
MAX_LAG = 5
CORR_THRESHOLD = 0.3
SIGNIFICANCE = 0.05
PAGERANK_ALPHA = 0.85
ANOMALY_WEIGHT = 0.4

METRIC_QUERIES = {
    "request_rate": 'rate(calls_total[1m])',
    "error_rate":   'rate(calls_total{status_code="STATUS_CODE_ERROR"}[1m]) / rate(calls_total[1m])',
    "latency":      'rate(duration_milliseconds_sum[1m]) / rate(duration_milliseconds_count[1m])',
}


# ── 1. PROMETHEUS FETCHER ────────────────────
def _query_prom(query, minutes_back=15, step="15s"):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes_back)
    try:
        r = requests.get(f"{PROM}/api/v1/query_range", params={
            "query": query, "start": start.isoformat(),
            "end": now.isoformat(), "step": step,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["data"]["result"] if data.get("status") == "success" else None
    except Exception as e:
        print(f"❌ Prometheus error: {e}")
        return None


def _to_dataframe(results):
    if not results:
        return pd.DataFrame()
    series_dict, ref_len = {}, None
    for item in results:
        svc = item["metric"].get("service_name") or item["metric"].get("job") or "unknown"
        vals = [float(v[1]) if v[1] != "NaN" else 0.0 for v in item.get("values", [])]
        if not vals:
            continue
        if ref_len is None:
            ref_len = len(vals)
        # align length
        vals = (vals + [vals[-1]] * max(0, ref_len - len(vals)))[:ref_len]
        series_dict[svc] = [a + b for a, b in zip(series_dict.get(svc, [0]*ref_len), vals)]
    if not series_dict:
        return pd.DataFrame()
    return pd.DataFrame(series_dict).ffill().fillna(0).replace([np.inf, -np.inf], 0)


def fetch_metrics(minutes_back=15, step="15s"):
    """Fetch request_rate, error_rate, latency from Prometheus → dict of DataFrames."""
    print(f"\n📡 Fetching metrics from Prometheus ({minutes_back}min, step={step})")
    metrics = {}
    for name, query in METRIC_QUERIES.items():
        results = _query_prom(query, minutes_back, step)
        df = _to_dataframe(results) if results else pd.DataFrame()
        if not df.empty:
            metrics[name] = df
            print(f"  ✅ {name}: {df.shape[1]} services, {df.shape[0]} steps")
        else:
            print(f"  ⚠  {name}: no data")
    if not metrics:
        print("❌ No metrics available!")
        return None
    # save CSV
    os.makedirs(LOG_DIR, exist_ok=True)
    if "request_rate" in metrics:
        metrics["request_rate"].to_csv(os.path.join(LOG_DIR, "metric_prometheus.csv"), index=False)
    return metrics


# ── 2. Z-SCORE ANOMALY DETECTION ────────────
def detect_anomalies(metrics, z_threshold=Z_THRESHOLD, window=ROLLING_WINDOW):
    """Z-score anomaly detection per service per metric."""
    all_svcs = sorted(set(c for df in metrics.values() for c in df.columns))
    results = {}
    for svc in all_svcs:
        z_scores, anom_list = {}, []
        for mname, df in metrics.items():
            if svc not in df.columns:
                continue
            s = df[svc].values
            if len(s) < window + 1:
                mu, sigma = np.mean(s), np.std(s)
            else:
                roll = pd.Series(s).rolling(window=window, min_periods=1)
                mu, sigma = roll.mean().iloc[-1], roll.std().fillna(0).iloc[-1]
            z = abs((s[-1] - mu) / sigma) if sigma > 1e-9 else 0.0
            z_scores[mname] = round(float(z), 4)
            if z > z_threshold:
                anom_list.append(mname)
        max_z = max(z_scores.values()) if z_scores else 0.0
        results[svc] = {
            "is_anomalous": len(anom_list) > 0,
            "max_z_score": round(max_z, 4),
            "anomaly_metrics": anom_list,
            "z_scores": z_scores,
            "severity": round(min(1.0, max_z / (z_threshold * 3)), 4),
        }
    # summary
    anom = [s for s, v in results.items() if v["is_anomalous"]]
    print(f"\n🔬 Anomaly detection: {len(anom)}/{len(results)} anomalous (threshold={z_threshold})")
    for s in all_svcs:
        v = results[s]
        flag = "🔴" if v["is_anomalous"] else "🟢"
        print(f"  {flag} {s:<20} z={v['max_z_score']:.2f}  severity={v['severity']:.2f}")
    return results


# ── 3. CAUSAL GRAPH (CROSS-CORRELATION) ─────
def _lagged_corr(x, y, max_lag=MAX_LAG):
    """Best lagged Pearson correlation corr(x(t), y(t+lag)). Returns (lag, corr, pval)."""
    n = len(x)
    best = (0, 0.0, 1.0)
    for lag in range(1, min(max_lag + 1, n - 2)):
        xs, ys = x[:n-lag], y[lag:]
        if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            continue
        c, p = sp_stats.pearsonr(xs, ys)
        if abs(c) > abs(best[1]):
            best = (lag, c, p)
    return best


def build_causal_graph(metrics, max_lag=MAX_LAG, corr_thr=CORR_THRESHOLD, sig=SIGNIFICANCE):
    """Build causal DAG from lagged cross-correlation across metrics."""
    primary = next((metrics[k] for k in ["request_rate", "latency", "error_rate"] if k in metrics), None)
    if primary is None:
        return nx.DiGraph()
    svcs = sorted(primary.columns)
    G = nx.DiGraph()
    G.add_nodes_from(svcs)
    if len(svcs) < 2:
        return G

    edges = []
    for a in svcs:
        for b in svcs:
            if a == b:
                continue
            corrs = []
            for mname, df in metrics.items():
                if a not in df.columns or b not in df.columns:
                    continue
                lag, c, p = _lagged_corr(df[a].values.astype(float), df[b].values.astype(float), max_lag)
                if lag > 0 and abs(c) > corr_thr and p < sig:
                    corrs.append({"metric": mname, "lag": lag, "corr": round(float(c), 4), "pval": round(float(p), 6)})
            if corrs:
                best = max(corrs, key=lambda x: abs(x["corr"]))
                w = round(float(np.mean([abs(c["corr"]) for c in corrs])), 4)
                edges.append((a, b, w, best))

    for a, b, w, best in sorted(edges, key=lambda e: e[2], reverse=True):
        G.add_edge(a, b, weight=w, best_corr=best["corr"], best_lag=best["lag"], best_metric=best["metric"])

    # enforce DAG
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda e: G[e[0]][e[1]].get("weight", 0))
        G.remove_edge(weakest[0], weakest[1])

    print(f"\n🔗 Causal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, DAG={nx.is_directed_acyclic_graph(G)}")
    for u, v, d in G.edges(data=True):
        print(f"  {u} → {v}  corr={d.get('best_corr', 0):.3f} lag={d.get('best_lag', 0)}")
    return G


# ── 4. PERSONALIZED PAGERANK RANKING ────────
def rank_root_causes(G, anomaly_info, alpha=PAGERANK_ALPHA, anom_w=ANOMALY_WEIGHT):
    """Personalized PageRank on reversed DAG + anomaly severity → ranked list."""
    if G.number_of_nodes() == 0:
        return []
    nodes = list(G.nodes())

    # personalization: anomalous nodes get higher seed
    pers = {}
    for n in nodes:
        info = anomaly_info.get(n, {})
        pers[n] = (info.get("severity", 0.0) + 0.1) if info.get("is_anomalous") else 0.01
    total = sum(pers.values())
    pers = {k: v / total for k, v in pers.items()}

    G_rev = G.reverse()
    try:
        pr = nx.pagerank(G_rev, alpha=alpha, personalization=pers, max_iter=200, weight="weight")
    except Exception:
        pr = {n: 1.0 / len(nodes) for n in nodes}
    try:
        rw = nx.pagerank(G_rev, alpha=0.95, personalization=pers, max_iter=200, weight="weight")
    except Exception:
        rw = pr.copy()

    max_pr = max(pr.values()) or 1
    max_rw = max(rw.values()) or 1

    ranking = []
    for n in nodes:
        info = anomaly_info.get(n, {})
        sev = info.get("severity", 0.0)
        pr_n = pr.get(n, 0) / max_pr
        rw_n = rw.get(n, 0) / max_rw
        score = (1 - anom_w) * (pr_n + rw_n) / 2 + anom_w * sev
        ranking.append((n, {
            "score": round(score, 6), "pagerank_norm": round(pr_n, 4), "rw_norm": round(rw_n, 4),
            "severity": sev, "is_anomalous": info.get("is_anomalous", False),
            "max_z_score": info.get("max_z_score", 0), "anomaly_metrics": info.get("anomaly_metrics", []),
            "causal_influence": G.out_degree(n) - G.in_degree(n),
            "out_degree": G.out_degree(n), "in_degree": G.in_degree(n),
        }))
    ranking.sort(key=lambda x: x[1]["score"], reverse=True)

    print(f"\n🏆 Root Cause Ranking:")
    for i, (n, v) in enumerate(ranking, 1):
        flag = "🔴" if v["is_anomalous"] else "🟢"
        rc = " ◀ ROOT CAUSE" if i == 1 else ""
        print(f"  {i}. {flag} {n:<20} score={v['score']:.4f} pr={v['pagerank_norm']:.3f} z={v['max_z_score']:.2f}{rc}")
    return ranking


# ── 5. PIPELINE ─────────────────────────────
def run_pipeline(minutes_back=15, step="15s"):
    print("\n" + "█" * 50)
    print("  GALA METRIC RCA PIPELINE")
    print("█" * 50)

    metrics = fetch_metrics(minutes_back, step)
    if not metrics:
        print(f"\n💡 Check Prometheus at {PROM}, ensure 'calls_total' metric exists.")
        return {"error": "No metrics"}

    anomalies = detect_anomalies(metrics)
    graph = build_causal_graph(metrics)
    ranking = rank_root_causes(graph, anomalies)

    # save JSON
    os.makedirs(LOG_DIR, exist_ok=True)
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": "GALA-metric-RCA",
        "ranking": [{"rank": i+1, "service": n, **v} for i, (n, v) in enumerate(ranking)],
        "anomalies": anomalies,
        "graph": {
            "nodes": list(graph.nodes()),
            "edges": [(u, v, d) for u, v, d in graph.edges(data=True)],
            "is_dag": nx.is_directed_acyclic_graph(graph),
        },
    }
    path = os.path.join(LOG_DIR, "metric_rca_ranking.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n✅ Done → {path}")
    if ranking:
        rc, info = ranking[0]
        print(f"🔥 Root cause: {rc} (score={info['score']:.4f}, z={info['max_z_score']:.2f})")
    return {"metrics": metrics, "anomalies": anomalies, "graph": graph, "ranking": ranking}


if __name__ == "__main__":
    run_pipeline(minutes_back=15, step="15s")