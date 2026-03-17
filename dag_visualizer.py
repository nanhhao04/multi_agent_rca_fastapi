"""
Jaeger Trace DAG Visualizer + Bayesian RCA
===========================================
1. Fetch traces from Jaeger API
2. Build NetworkX DiGraph from spans (parent → child)
3. Bayesian Network (pgmpy) → P(root_cause | observed_anomalies)
4. Export to HTML with D3.js interactive visualization
"""

import json
import os
import statistics
import webbrowser
from collections import defaultdict

import numpy as np
import requests
import networkx as nx
from tabulate import tabulate

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ──────────────────────────────────────────────
# 1. JAEGER TRACE FETCHER
# ──────────────────────────────────────────────
JAEGER_URL = "http://localhost:16686"


def fetch_services():
    resp = requests.get(f"{JAEGER_URL}/api/services")
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_traces(service="app-a", operation=None, limit=20):
    params = {"service": service, "limit": limit}
    if operation:
        params["operation"] = operation
    resp = requests.get(f"{JAEGER_URL}/api/traces", params=params)
    resp.raise_for_status()
    return resp.json()["data"]


# ──────────────────────────────────────────────
# 2. DAG BUILDER (NetworkX)
# ──────────────────────────────────────────────
def build_dag_from_trace(trace: dict) -> nx.DiGraph:
    """Build span-level DAG from a single Jaeger trace."""
    G = nx.DiGraph()
    spans = {s["spanID"]: s for s in trace["spans"]}
    processes = trace["processes"]

    for span_id, span in spans.items():
        service = processes[span["processID"]]["serviceName"]
        op = span["operationName"]
        duration_ms = span["duration"] / 1000

        tags = {t["key"]: t["value"] for t in span.get("tags", [])}
        http_status = tags.get("http.status_code", 200)
        has_error = tags.get("error", False) or (isinstance(http_status, int) and http_status >= 500)

        G.add_node(span_id,
                   service=service,
                   operation=op,
                   duration_ms=round(duration_ms, 2),
                   has_error=has_error,
                   start_time=span["startTime"])

        for ref in span.get("references", []):
            if ref["refType"] == "CHILD_OF":
                parent_id = ref["spanID"]
                G.add_edge(parent_id, span_id)

    return G


def build_merged_service_dag(traces: list) -> tuple:
    """
    Merge multiple traces → service-level DAG.
    Group spans by label (service::operation), compute stats.

    Returns:
        (dag, stats) where stats[label] = { durations, error_count, total_count, ... }
    """
    stats = defaultdict(lambda: {
        "durations": [],
        "error_count": 0,
        "total_count": 0,
        "service_name": "",
        "operation": "",
    })
    edge_set = set()

    for trace in traces:
        spans = {s["spanID"]: s for s in trace["spans"]}
        processes = trace["processes"]
        span_id_to_label = {}

        for span_id, span in spans.items():
            svc = processes[span["processID"]]["serviceName"]
            op = span["operationName"]
            label = f"{svc}::{op}"
            span_id_to_label[span_id] = label
            dur = span["duration"] / 1000

            stats[label]["durations"].append(dur)
            stats[label]["total_count"] += 1
            stats[label]["service_name"] = svc
            stats[label]["operation"] = op

            tags = {t["key"]: t["value"] for t in span.get("tags", [])}
            http_status = tags.get("http.status_code", 200)
            has_error = tags.get("error", False) or (isinstance(http_status, int) and http_status >= 500)
            if has_error:
                stats[label]["error_count"] += 1

        for span_id, span in spans.items():
            child_label = span_id_to_label.get(span_id)
            for ref in span.get("references", []):
                if ref["refType"] == "CHILD_OF":
                    parent_label = span_id_to_label.get(ref["spanID"])
                    if parent_label and child_label and parent_label != child_label:
                        edge_set.add((parent_label, child_label))

    G = nx.DiGraph()
    for label, st in stats.items():
        durations = st["durations"]
        mean_dur = statistics.mean(durations) if durations else 0
        std_dur = statistics.stdev(durations) if len(durations) > 1 else 0
        error_rate = st["error_count"] / st["total_count"] if st["total_count"] > 0 else 0

        G.add_node(label, **{
            "service_name": st["service_name"],
            "operation": st["operation"],
            "mean_duration_ms": round(mean_dur, 2),
            "std_duration_ms": round(std_dur, 2),
            "error_rate": round(error_rate, 4),
            "total_count": st["total_count"],
            "error_count": st["error_count"],
        })

    for parent, child in edge_set:
        if parent in G and child in G:
            G.add_edge(parent, child)

    return G, dict(stats)


# ──────────────────────────────────────────────
# 3. ANOMALY DETECTION (z-score)
# ──────────────────────────────────────────────
def detect_anomalies(trace: dict, stats: dict, z_threshold: float = 2.0) -> dict:
    """
    Detect anomalies in a single trace using z-score on duration.
    Returns: dict[label] → is_anomalous (bool)
    """
    spans = trace["spans"]
    processes = trace["processes"]
    anomalies = {}

    for span in spans:
        svc = processes[span["processID"]]["serviceName"]
        op = span["operationName"]
        label = f"{svc}::{op}"
        dur = span["duration"] / 1000
        is_anomalous = False

        # Check error tags
        tags = {t["key"]: t["value"] for t in span.get("tags", [])}
        http_status = tags.get("http.status_code", 200)
        if tags.get("error", False) or (isinstance(http_status, int) and http_status >= 500):
            is_anomalous = True

        # Check duration z-score
        if label in stats:
            durs = stats[label]["durations"]
            if len(durs) > 1:
                mean = statistics.mean(durs)
                std = statistics.stdev(durs)
                if std > 0 and dur > mean + z_threshold * std:
                    is_anomalous = True

        anomalies[label] = anomalies.get(label, False) or is_anomalous

    return anomalies


# ──────────────────────────────────────────────
# 4. BAYESIAN RCA (pgmpy)
# ──────────────────────────────────────────────
class BayesianRCA:
    """
    Bayesian Network for Root Cause Analysis.
    
    Bayes' Theorem:
        P(root_cause | observed_anomaly) ∝ P(observed_anomaly | root_cause) × P(root_cause)
    
    Each node is binary: 0=normal, 1=anomalous.
    CPD learned from trace data:
        - Root nodes: P(anomalous) = error_rate from data
        - Child nodes: P(child_anomalous | parent_states) = base_rate + prop_factor × fraction_parents_anomalous
    """

    def __init__(self):
        self.model = None
        self.dag = None
        self.node_labels = []
        self.inference_engine = None
        self._name_map = {}
        self._reverse_map = {}

    def build_from_dag(self, dag: nx.DiGraph, stats: dict, traces: list,
                       base_anomaly_rate=0.05, propagation_factor=0.7):
        """Build Bayesian Network from merged service DAG."""
        self.dag = dag
        self.node_labels = list(dag.nodes())

        if not self.node_labels:
            print("[BayesianRCA] Empty DAG.")
            return

        # Sanitize node names for pgmpy
        for i, label in enumerate(self.node_labels):
            safe = f"N{i}"
            self._name_map[label] = safe
            self._reverse_map[safe] = label

        # Build pgmpy model
        edges = [(self._name_map[p], self._name_map[c]) for p, c in dag.edges()]
        self.model = DiscreteBayesianNetwork(edges)

        for label in self.node_labels:
            safe = self._name_map[label]
            if safe not in self.model.nodes():
                self.model.add_node(safe)

        # Compute anomaly rates from data
        anomaly_rates = self._compute_anomaly_rates(traces, stats)

        # Create CPD for each node
        for label in self.node_labels:
            safe = self._name_map[label]
            parents = list(self.model.get_parents(safe))

            if not parents:
                # Root node: P(anomalous) = error_rate
                rate = max(0.001, min(0.999, anomaly_rates.get(label, base_anomaly_rate)))
                cpd = TabularCPD(
                    variable=safe, variable_card=2,
                    values=[[1 - rate], [rate]],
                    state_names={safe: ["normal", "anomalous"]},
                )
            else:
                # Child node: P(anomalous | parents)
                n_parents = len(parents)
                node_rate = max(0.001, min(0.999, anomaly_rates.get(label, base_anomaly_rate)))
                vals_normal, vals_anomalous = [], []

                for combo in range(2 ** n_parents):
                    n_anom = bin(combo).count("1")
                    frac = n_anom / n_parents
                    p_anom = node_rate + (1 - node_rate) * frac * propagation_factor
                    p_anom = max(0.001, min(0.999, p_anom))
                    vals_normal.append(1 - p_anom)
                    vals_anomalous.append(p_anom)

                cpd = TabularCPD(
                    variable=safe, variable_card=2,
                    values=[vals_normal, vals_anomalous],
                    evidence=parents, evidence_card=[2] * n_parents,
                    state_names={safe: ["normal", "anomalous"],
                                 **{p: ["normal", "anomalous"] for p in parents}},
                )

            self.model.add_cpds(cpd)

        valid = self.model.check_model()
        print(f"[BayesianRCA] Model {'OK' if valid else 'INVALID'} — "
              f"{len(self.node_labels)} nodes, {len(edges)} edges")

        self.inference_engine = VariableElimination(self.model)

    def _compute_anomaly_rates(self, traces, stats, z_threshold=2.0):
        counts = defaultdict(int)
        totals = defaultdict(int)
        for trace in traces:
            anoms = detect_anomalies(trace, stats, z_threshold)
            for label, is_anom in anoms.items():
                totals[label] += 1
                if is_anom:
                    counts[label] += 1
        result = {}
        for label in self.node_labels:
            if totals[label] > 0:
                result[label] = counts[label] / totals[label]
            else:
                st = stats.get(label, {})
                result[label] = st.get("error_count", 0) / max(st.get("total_count", 1), 1)
        return result

    def run_rca(self, evidence: dict, top_k=20) -> list:
        """
        Run RCA inference.
        
        evidence: dict[label] → "anomalous"
        Returns: list sorted by P(anomalous | evidence)
        """
        if not self.inference_engine:
            return []

        safe_evidence = {}
        for label, state in evidence.items():
            if label in self._name_map:
                safe_evidence[self._name_map[label]] = state

        results = []
        query_nodes = [self._name_map[l] for l in self.node_labels if l not in evidence]

        for safe in query_nodes:
            try:
                result = self.inference_engine.query(variables=[safe], evidence=safe_evidence)
                p_anom = float(result.values[1])
                label = self._reverse_map[safe]
                data = self.dag.nodes[label]
                results.append({
                    "node": label,
                    "service": data.get("service_name", ""),
                    "operation": data.get("operation", ""),
                    "posterior": round(p_anom, 4),
                    "error_rate": data.get("error_rate", 0),
                    "mean_duration_ms": data.get("mean_duration_ms", 0),
                })
            except Exception as e:
                print(f"[BayesianRCA] Query error for {safe}: {e}")

        # Add evidence nodes with posterior=1.0
        for label in evidence:
            if label in self._name_map:
                data = self.dag.nodes[label]
                results.append({
                    "node": label,
                    "service": data.get("service_name", ""),
                    "operation": data.get("operation", ""),
                    "posterior": 1.0,
                    "error_rate": data.get("error_rate", 0),
                    "mean_duration_ms": data.get("mean_duration_ms", 0),
                    "is_evidence": True,
                })

        results.sort(key=lambda x: x["posterior"], reverse=True)
        return results[:top_k]

    def run_rca_from_trace(self, trace, stats, z_threshold=2.0, top_k=10):
        """Auto-detect anomalies in a trace, then run RCA."""
        anomalies = detect_anomalies(trace, stats, z_threshold)
        evidence = {l: "anomalous" for l, is_anom in anomalies.items()
                    if is_anom and l in self._name_map}
        if not evidence:
            print("[BayesianRCA] No anomalies detected in this trace.")
            return []
        print(f"[BayesianRCA] {len(evidence)} anomalous nodes → running inference...")
        return self.run_rca(evidence, top_k)


# ──────────────────────────────────────────────
# 5. CONSOLE OUTPUT
# ──────────────────────────────────────────────
def print_dag_info(G, title="DAG"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}  |  "
          f"DAG: {nx.is_directed_acyclic_graph(G)}")
    if nx.is_directed_acyclic_graph(G) and G.number_of_nodes() > 0:
        print(f"  Longest path: {nx.dag_longest_path_length(G)}")
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]
        print(f"  Roots: {roots}")
        print(f"  Leaves: {leaves}")
    print(f"\n  {'Node':<45} {'Err%':>6} {'Avg(ms)':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*8}")
    for node, data in G.nodes(data=True):
        err = data.get("error_rate", 0) * 100
        dur = data.get("mean_duration_ms", 0)
        print(f"  {node:<45} {err:>5.1f}% {dur:>7.1f}")
    print(f"\n  Edges:")
    for p, c in G.edges():
        print(f"    {p} → {c}")
    print()


def print_rca_results(results, title="RCA Results"):
    print(f"\n{'='*60}")
    print(f"  🔍 {title}")
    print(f"{'='*60}")
    if not results:
        print("  No anomalies detected.")
        return
    table = []
    for i, r in enumerate(results, 1):
        table.append([
            i, r["service"], r["operation"],
            f"{r['posterior']:.2%}",
            f"{r['error_rate']:.2%}",
            f"{r['mean_duration_ms']:.1f}ms",
            "⚡" if r.get("is_evidence") else "",
        ])
    headers = ["#", "Service", "Operation", "P(Root Cause)", "Error Rate", "Avg Dur", "Ev"]
    print(tabulate(table, headers=headers, tablefmt="rounded_grid"))
    print()


# ──────────────────────────────────────────────
# 6. HTML GENERATION
# ──────────────────────────────────────────────
def dag_to_json(G, rca_results=None):
    """Convert NetworkX DAG + RCA results to JSON for D3.js."""
    nodes = []
    # Build posterior lookup
    posterior_map = {}
    if rca_results:
        for r in rca_results:
            posterior_map[r["node"]] = r

    for nid, data in G.nodes(data=True):
        rca = posterior_map.get(nid, {})
        nodes.append({
            "id": nid,
            "service": data.get("service_name", nid),
            "operation": data.get("operation", ""),
            "mean_duration_ms": data.get("mean_duration_ms", 0),
            "std_duration_ms": data.get("std_duration_ms", 0),
            "error_rate": data.get("error_rate", 0),
            "total_count": data.get("total_count", 0),
            "in_degree": G.in_degree(nid),
            "out_degree": G.out_degree(nid),
            "posterior": rca.get("posterior", 0),
            "is_evidence": rca.get("is_evidence", False),
        })

    edges = [{"source": u, "target": v} for u, v in G.edges()]

    return {
        "nodes": nodes,
        "edges": edges,
        "is_dag": nx.is_directed_acyclic_graph(G),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "max_depth": nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) and G.number_of_nodes() > 0 else 0,
    }


def generate_html(graph_data: dict, output_path="dag_rca_output.html"):
    graph_json = json.dumps(graph_data, indent=None)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Jaeger DAG + Bayesian RCA</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#0a0a18;--bg2:#12122a;--bg3:#1a1a3a;--card:#1e1e42;
      --border:rgba(124,92,252,.12);
      --accent:#7c5cfc;--accent2:#00e5a0;--accent3:#3dadff;
      --glow:rgba(124,92,252,.3);--glow2:rgba(0,229,160,.25);
      --text:#e8e8f8;--text2:#9898c8;
      --danger:#ff4d6a;--warn:#ffb133;--success:#00e5a0;
      --mono:'JetBrains Mono',monospace;--sans:'Inter',system-ui,sans-serif;
      --radius:14px;
    }
    body{font-family:var(--sans);background:var(--bg);color:var(--text);height:100vh;overflow:hidden}
    .app{display:grid;grid-template-columns:380px 1fr;grid-template-rows:58px 1fr;height:100vh}

    /* Header */
    header{grid-column:1/-1;display:flex;align-items:center;gap:14px;padding:0 24px;
      background:linear-gradient(135deg,var(--bg2),var(--bg));border-bottom:1px solid var(--border)}
    .logo{width:34px;height:34px;border-radius:10px;
      background:linear-gradient(135deg,var(--accent),var(--accent2));
      display:grid;place-items:center;font-weight:800;font-size:16px;color:#fff}
    header h1{font-size:16px;font-weight:700;
      background:linear-gradient(90deg,var(--accent),var(--accent2));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .tag{font-size:9px;font-weight:700;padding:3px 10px;border-radius:20px;
      background:rgba(0,229,160,.12);color:var(--accent2);letter-spacing:.8px;text-transform:uppercase}
    .hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
    .dag-badge{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;
      padding:4px 12px;border-radius:20px}
    .dag-badge.ok{background:rgba(0,229,160,.1);color:var(--success)}
    .dag-badge.fail{background:rgba(255,77,106,.1);color:var(--danger)}

    /* Sidebar */
    .sidebar{background:var(--bg2);border-right:1px solid var(--border);overflow-y:auto;padding:16px;
      display:flex;flex-direction:column;gap:16px}
    .sidebar::-webkit-scrollbar{width:3px}
    .sidebar::-webkit-scrollbar-thumb{background:var(--accent);border-radius:3px}
    .sec{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:1.8px;color:var(--text2);margin-bottom:4px}

    /* Stats */
    .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
    .st{background:var(--card);border-radius:10px;padding:10px 8px;text-align:center;border:1px solid rgba(255,255,255,.02)}
    .st .v{font-size:18px;font-weight:800;
      background:linear-gradient(135deg,var(--accent),var(--accent2));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .st .l{font-size:8px;color:var(--text2);margin-top:2px;text-transform:uppercase;letter-spacing:.4px}

    /* RCA Table */
    .rca-table{width:100%;border-collapse:collapse;font-size:11px}
    .rca-table th{text-align:left;color:var(--text2);font-size:9px;font-weight:700;
      text-transform:uppercase;letter-spacing:.8px;padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.06)}
    .rca-table td{padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.03)}
    .rca-table tr{cursor:pointer;transition:background .15s}
    .rca-table tbody tr:hover{background:rgba(124,92,252,.06)}
    .rca-bar{height:6px;border-radius:3px;min-width:2px;transition:width .5s}
    .rca-rank{width:20px;text-align:center;font-weight:700;color:var(--text2);font-family:var(--mono)}
    .rca-post{font-family:var(--mono);font-weight:600;font-size:11px}
    .ev-badge{font-size:8px;font-weight:700;padding:1px 5px;border-radius:6px;
      background:rgba(255,77,106,.15);color:var(--danger);letter-spacing:.3px}

    /* Node list */
    .nlist{display:flex;flex-direction:column;gap:4px}
    .nitem{display:flex;align-items:center;gap:8px;background:var(--card);border-radius:10px;
      padding:8px 10px;cursor:pointer;border:1px solid transparent;transition:all .2s;font-size:11px}
    .nitem:hover,.nitem.active{border-color:var(--accent);box-shadow:0 0 12px var(--glow)}
    .ndot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .nitem .nm{font-weight:500;flex:1}
    .nitem .dr{color:var(--text2);font-family:var(--mono);font-size:10px}

    /* Buttons */
    .btn{display:flex;align-items:center;justify-content:center;gap:5px;padding:8px;border:none;border-radius:10px;
      font-family:var(--sans);font-size:11px;font-weight:600;cursor:pointer;transition:all .2s}
    .btn-s{background:var(--card);color:var(--text);border:1px solid rgba(255,255,255,.04)}
    .btn-s:hover{border-color:var(--accent)}

    /* Graph */
    .gc{position:relative;overflow:hidden;
      background:radial-gradient(circle at 25% 45%,rgba(124,92,252,.05) 0%,transparent 50%),
        radial-gradient(circle at 75% 35%,rgba(0,229,160,.04) 0%,transparent 50%),var(--bg)}
    svg{width:100%;height:100%}
    .edge{fill:none;stroke-width:2;opacity:.5;transition:opacity .3s,stroke-width .3s}
    .edge:hover,.edge.hl{stroke-width:3.5;opacity:1}
    @keyframes dash{to{stroke-dashoffset:-20}}
    .edge-anim{stroke-dasharray:6 4;animation:dash .8s linear infinite}
    .node{cursor:grab}.node:active{cursor:grabbing}
    .node-circle{stroke-width:2;transition:filter .2s}
    .node:hover .node-circle{filter:drop-shadow(0 0 14px var(--glow))}
    .node-label{font-family:var(--sans);font-size:10px;font-weight:700;fill:var(--text);text-anchor:middle;pointer-events:none}

    /* Tooltip */
    .tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid rgba(124,92,252,.25);
      border-radius:var(--radius);padding:14px;min-width:250px;box-shadow:0 14px 44px rgba(0,0,0,.55);
      opacity:0;transform:translateY(6px);transition:opacity .15s,transform .15s;z-index:100}
    .tip.show{opacity:1;transform:translateY(0)}
    .tip h3{font-size:13px;font-weight:700;margin-bottom:8px;color:var(--accent)}
    .tip .r{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;
      border-bottom:1px solid rgba(255,255,255,.03)}
    .tip .r:last-child{border-bottom:none}
    .tip .k{color:var(--text2)}
    .tip .vl{font-weight:600;font-family:var(--mono);font-size:10px}

    .zoom-ind{position:absolute;top:12px;right:14px;background:rgba(26,26,58,.85);backdrop-filter:blur(8px);
      padding:4px 10px;border-radius:8px;font-size:10px;color:var(--text2);border:1px solid rgba(255,255,255,.04)}
    .legend{position:absolute;bottom:14px;right:14px;background:rgba(26,26,58,.85);backdrop-filter:blur(10px);
      border:1px solid var(--border);border-radius:var(--radius);padding:10px 14px;font-size:9px;color:var(--text2)}
    .legend-r{display:flex;align-items:center;gap:6px;padding:2px 0}
    .legend-d{width:8px;height:8px;border-radius:50%}

    /* Color scale for posterior */
    .post-high{color:#ff4d6a}
    .post-med{color:#ffb133}
    .post-low{color:#00e5a0}
  </style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo">B</div>
    <h1>Bayesian RCA — Jaeger Traces</h1>
    <span class="tag">P(root cause | anomaly)</span>
    <div class="hdr-right">
      <div class="dag-badge" id="dag-badge"><span>—</span></div>
    </div>
  </header>

  <aside class="sidebar">
    <div>
      <div class="sec">Graph Stats</div>
      <div class="stats">
        <div class="st"><div class="v" id="s-n">0</div><div class="l">Nodes</div></div>
        <div class="st"><div class="v" id="s-e">0</div><div class="l">Edges</div></div>
        <div class="st"><div class="v" id="s-d">0</div><div class="l">Depth</div></div>
        <div class="st"><div class="v" id="s-r">0</div><div class="l">Roots</div></div>
      </div>
    </div>

    <div>
      <div class="sec">Bayesian RCA — Root Cause Ranking</div>
      <table class="rca-table" id="rca-table">
        <thead><tr><th>#</th><th>Service</th><th>Operation</th><th>P(RC)</th><th></th></tr></thead>
        <tbody id="rca-body"></tbody>
      </table>
    </div>

    <div>
      <div class="sec">Bayes Formula</div>
      <div style="background:var(--card);border-radius:var(--radius);padding:12px;font-size:11px;line-height:1.7;border:1px solid rgba(255,255,255,.03)">
        <div style="font-family:var(--mono);color:var(--accent);font-size:12px;margin-bottom:6px">
          P(Xᵢ=cause | E) ∝ P(E | Xᵢ=cause) × P(Xᵢ)
        </div>
        <div style="color:var(--text2)">
          <strong style="color:var(--text)">Prior</strong> P(Xᵢ) = error_rate từ traces<br>
          <strong style="color:var(--text)">Likelihood</strong> P(E|Xᵢ) = propagation qua DAG<br>
          <strong style="color:var(--text)">Posterior</strong> = Variable Elimination (pgmpy)
        </div>
      </div>
    </div>

    <div>
      <div class="sec">All Nodes</div>
      <div class="nlist" id="nlist"></div>
    </div>

    <div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <button class="btn btn-s" id="btn-reset">↻ Reset Layout</button>
        <button class="btn btn-s" id="btn-anim">◎ Toggle Animation</button>
      </div>
    </div>
  </aside>

  <div class="gc" id="gc">
    <svg id="svg"></svg>
    <div class="tip" id="tip"></div>
    <div class="zoom-ind" id="zi">100%</div>
    <div class="legend">
      <div class="legend-r"><div class="legend-d" style="background:var(--accent)"></div> Root (no parent)</div>
      <div class="legend-r"><div class="legend-d" style="background:var(--accent2)"></div> Internal</div>
      <div class="legend-r"><div class="legend-d" style="background:var(--warn)"></div> Leaf</div>
      <div class="legend-r"><div class="legend-d" style="background:var(--danger)"></div> High P(root cause)</div>
      <div class="legend-r" style="margin-top:3px;opacity:.6">Node size ∝ posterior probability</div>
    </div>
  </div>
</div>
<script>
const DATA = """ + graph_json + """;

const PAL=['#7c5cfc','#00e5a0','#3dadff','#ff922b','#e599f7','#69db7c','#ffd43b','#ff4d6a','#38d9a9','#da77f2','#66d9e8','#f783ac'];
const scm=new Map(); let ci=0;
function sc(s){if(!scm.has(s))scm.set(s,PAL[ci++%PAL.length]);return scm.get(s)}

function postColor(p){
  if(p>=0.7) return '#ff4d6a';
  if(p>=0.4) return '#ffb133';
  if(p>=0.1) return '#3dadff';
  return '#00e5a0';
}

function nType(n){return n.in_degree===0?'root':n.out_degree===0?'leaf':'internal'}
function nRadius(d){
  // Size by posterior (bigger = more likely root cause)
  const base = 16;
  const postBonus = d.posterior * 28;
  return Math.max(base, Math.min(44, base + postBonus));
}
function fmtDur(ms){return ms>=1000?(ms/1000).toFixed(2)+'s':ms.toFixed(1)+'ms'}

// D3
const svg=d3.select('#svg'), gc=document.getElementById('gc'), tip=document.getElementById('tip');
let anim=true, W, H, sim;
function dims(){const r=gc.getBoundingClientRect();W=r.width;H=r.height;svg.attr('viewBox',`0 0 ${W} ${H}`)}
dims(); window.addEventListener('resize',dims);

svg.append('defs').append('marker').attr('id','arr').attr('viewBox','0 -5 10 10')
  .attr('refX',28).attr('refY',0).attr('markerWidth',7).attr('markerHeight',7).attr('orient','auto')
  .append('path').attr('d','M0,-5L10,0L0,5').attr('fill','var(--accent)');
const gr=svg.select('defs').append('linearGradient').attr('id','eg');
gr.append('stop').attr('offset','0%').attr('stop-color','#7c5cfc');
gr.append('stop').attr('offset','100%').attr('stop-color','#00e5a0');

const gM=svg.append('g');
const zm=d3.zoom().scaleExtent([.2,5]).on('zoom',e=>{
  gM.attr('transform',e.transform);
  document.getElementById('zi').textContent=Math.round(e.transform.k*100)+'%'});
svg.call(zm);
const eL=gM.append('g'), nL=gM.append('g');

function render(data){
  eL.selectAll('*').remove(); nL.selectAll('*').remove();
  const nodes=data.nodes.map(n=>({...n}));
  const edges=data.edges.map(e=>({...e}));
  if(!nodes.length) return;

  if(sim) sim.stop();
  sim=d3.forceSimulation(nodes)
    .force('link',d3.forceLink(edges).id(d=>d.id).distance(120).strength(.7))
    .force('charge',d3.forceManyBody().strength(-550))
    .force('center',d3.forceCenter(W/2,H/2))
    .force('y',d3.forceY(d=>{const t=nType(d);return t==='root'?H*.15:t==='leaf'?H*.85:H*.5}).strength(.12))
    .force('collision',d3.forceCollide().radius(d=>nRadius(d)+12));

  const eS=eL.selectAll('.edge').data(edges).enter().append('line')
    .attr('class','edge'+(anim?' edge-anim':'')).attr('stroke','url(#eg)').attr('marker-end','url(#arr)');

  const nS=nL.selectAll('.node').data(nodes).enter().append('g').attr('class','node');

  // Posterior halo (red glow for high-posterior nodes)
  nS.append('circle').attr('class','post-halo')
    .attr('r',d=>nRadius(d)+10)
    .attr('fill','none')
    .attr('stroke',d=>postColor(d.posterior))
    .attr('stroke-width',d=>d.posterior>0.3?2:1)
    .attr('opacity',d=>Math.min(.6, d.posterior*.8))
    .attr('stroke-dasharray',d=>d.is_evidence?'none':'4 2');

  // Main circle
  nS.append('circle').attr('class','node-circle')
    .attr('r',d=>nRadius(d))
    .attr('fill',d=>postColor(d.posterior))
    .attr('fill-opacity',d=>0.08+d.posterior*0.25)
    .attr('stroke',d=>postColor(d.posterior))
    .attr('stroke-width',2);

  // Service label
  nS.append('text').attr('class','node-label').attr('dy',d=>nRadius(d)+16).text(d=>d.service);

  // Operation label
  nS.append('text').attr('class','node-label')
    .attr('dy',d=>nRadius(d)+27).attr('font-size','8px').attr('font-weight','400').attr('fill','var(--text2)')
    .text(d=>{const o=d.operation;return o.length>22?o.slice(0,20)+'…':o});

  // Posterior inside node
  nS.append('text').attr('class','node-label')
    .attr('dy',2).attr('font-size','10px').attr('font-family','var(--mono)')
    .attr('fill',d=>postColor(d.posterior))
    .text(d=>d.posterior>0?(d.posterior*100).toFixed(1)+'%':'—');

  // Evidence badge
  nS.filter(d=>d.is_evidence).append('text')
    .attr('class','node-label').attr('dy',d=>-nRadius(d)-6)
    .attr('font-size','8px').attr('fill','var(--danger)').attr('font-weight','700')
    .text('⚡ EVIDENCE');

  // Drag
  nS.call(d3.drag()
    .on('start',(e,d)=>{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y})
    .on('drag',(e,d)=>{d.fx=e.x;d.fy=e.y})
    .on('end',(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));

  // Tooltip
  nS.on('mouseenter',(e,d)=>{
    tip.innerHTML=`
      <h3>${d.service}::${d.operation}</h3>
      <div class="r"><span class="k">P(Root Cause)</span><span class="vl" style="color:${postColor(d.posterior)}">${(d.posterior*100).toFixed(2)}%</span></div>
      <div class="r"><span class="k">Error Rate</span><span class="vl">${(d.error_rate*100).toFixed(2)}%</span></div>
      <div class="r"><span class="k">Avg Duration</span><span class="vl">${fmtDur(d.mean_duration_ms)}</span></div>
      <div class="r"><span class="k">Std Dev</span><span class="vl">${fmtDur(d.std_duration_ms)}</span></div>
      <div class="r"><span class="k">Samples</span><span class="vl">${d.total_count}</span></div>
      <div class="r"><span class="k">Type</span><span class="vl">${nType(d)}${d.is_evidence?' (evidence)':''}</span></div>
      <div class="r"><span class="k">In-degree</span><span class="vl">${d.in_degree}</span></div>
      <div class="r"><span class="k">Out-degree</span><span class="vl">${d.out_degree}</span></div>
    `;
    tip.classList.add('show');
    eS.classed('hl',l=>(l.source.id||l.source)===d.id||(l.target.id||l.target)===d.id);
  })
  .on('mousemove',e=>{const r=gc.getBoundingClientRect();tip.style.left=(e.clientX-r.left+12)+'px';tip.style.top=(e.clientY-r.top+12)+'px'})
  .on('mouseleave',()=>{tip.classList.remove('show');eS.classed('hl',false)});

  sim.on('tick',()=>{
    eS.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    nS.attr('transform',d=>`translate(${d.x},${d.y})`);
  });

  updateUI(data);
}

function updateUI(data){
  const roots=data.nodes.filter(n=>n.in_degree===0);
  document.getElementById('s-n').textContent=data.num_nodes;
  document.getElementById('s-e').textContent=data.num_edges;
  document.getElementById('s-d').textContent=data.max_depth;
  document.getElementById('s-r').textContent=roots.length;

  const b=document.getElementById('dag-badge');
  b.className='dag-badge '+(data.is_dag?'ok':'fail');
  b.querySelector('span').textContent=data.is_dag?'nx.is_directed_acyclic_graph(G) = True':'Cycle!';

  // RCA table
  const tbody=document.getElementById('rca-body');
  tbody.innerHTML='';
  const ranked=data.nodes.filter(n=>n.posterior>0).sort((a,b)=>b.posterior-a.posterior);
  ranked.forEach((n,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td class="rca-rank">${i+1}</td>
      <td><span style="color:${sc(n.service)}">${n.service}</span></td>
      <td style="font-family:var(--mono);font-size:10px;color:var(--text2)">${n.operation}</td>
      <td class="rca-post" style="color:${postColor(n.posterior)}">${(n.posterior*100).toFixed(1)}%</td>
      <td style="width:80px"><div class="rca-bar" style="width:${n.posterior*100}%;background:${postColor(n.posterior)}"></div></td>
    `;
    if(n.is_evidence) tr.querySelector('td:nth-child(2)').innerHTML+=` <span class="ev-badge">EV</span>`;
    tr.onclick=()=>highlightNode(n.id);
    tbody.appendChild(tr);
  });

  // Node list
  const nl=document.getElementById('nlist');
  nl.innerHTML='';
  data.nodes.sort((a,b)=>b.posterior-a.posterior).forEach(n=>{
    const el=document.createElement('div');
    el.className='nitem';
    el.innerHTML=`
      <div class="ndot" style="background:${postColor(n.posterior)}"></div>
      <span class="nm">${n.service}::${n.operation.slice(0,15)}</span>
      <span class="dr">${(n.posterior*100).toFixed(1)}%</span>
    `;
    el.onclick=()=>highlightNode(n.id);
    nl.appendChild(el);
  });
}

function highlightNode(id){
  nL.selectAll('.node').each(function(d){
    if(d.id===id){
      d3.select(this).select('.post-halo')
        .transition().duration(250).attr('opacity',.8).attr('r',nRadius(d)+18)
        .transition().duration(500).attr('opacity',Math.min(.6,d.posterior*.8)).attr('r',nRadius(d)+10);
      const t=d3.zoomTransform(svg.node());
      svg.transition().duration(500).call(zm.transform,
        d3.zoomIdentity.translate(W/2-d.x*t.k,H/2-d.y*t.k).scale(t.k));
    }
  });
}

document.getElementById('btn-reset').onclick=()=>{svg.transition().duration(500).call(zm.transform,d3.zoomIdentity);render(DATA)};
document.getElementById('btn-anim').onclick=()=>{anim=!anim;eL.selectAll('.edge').classed('edge-anim',anim)};

render(DATA);
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Jaeger Trace DAG + Bayesian RCA")
    print("  NetworkX → pgmpy → D3.js")
    print("=" * 60)

    # 1) Fetch services
    print("\n[1] Fetching services from Jaeger...")
    try:
        services = fetch_services()
        print(f"    Services: {services}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    # 2) Fetch traces
    service = "app-a"
    print(f"\n[2] Fetching traces for '{service}'...")
    try:
        traces = fetch_traces(service=service, limit=20)
        print(f"    Got {len(traces)} traces")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    if not traces:
        print("    No traces! Try: curl http://localhost:8000/chain")
        return

    # 3) Build merged service DAG
    print(f"\n[3] Building merged service DAG (NetworkX)...")
    dag, stats = build_merged_service_dag(traces)
    print_dag_info(dag, "Merged Service::Operation DAG")

    # Verify DAG
    print(f"    nx.is_directed_acyclic_graph(G) = {nx.is_directed_acyclic_graph(G=dag)}")

    # 4) Build Bayesian Network
    print(f"\n[4] Building Bayesian Network (pgmpy)...")
    rca = BayesianRCA()
    rca.build_from_dag(dag, stats, traces,
                       base_anomaly_rate=0.05,
                       propagation_factor=0.7)

    # 5) Run RCA on the latest trace
    print(f"\n[5] Running RCA on latest trace...")
    rca_results = rca.run_rca_from_trace(traces[0], stats, top_k=15)
    print_rca_results(rca_results, "P(Root Cause | Observed Anomalies)")

    # If no anomalies found, run RCA with all nodes as query (no evidence)
    if not rca_results:
        print("    No anomalies auto-detected → showing prior probabilities...")
        rca_results = []
        for label in rca.node_labels:
            safe = rca._name_map[label]
            try:
                result = rca.inference_engine.query(variables=[safe])
                p = float(result.values[1])
                data = dag.nodes[label]
                rca_results.append({
                    "node": label,
                    "service": data.get("service_name", ""),
                    "operation": data.get("operation", ""),
                    "posterior": round(p, 4),
                    "error_rate": data.get("error_rate", 0),
                    "mean_duration_ms": data.get("mean_duration_ms", 0),
                })
            except Exception:
                pass
        rca_results.sort(key=lambda x: x["posterior"], reverse=True)
        print_rca_results(rca_results, "Prior P(anomalous) — no evidence")

    # 6) Generate HTML
    graph_data = dag_to_json(dag, rca_results)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dag_rca_output.html")
    print(f"\n[6] Generating HTML...")
    generate_html(graph_data, output_path)
    print(f"    → {output_path}")

    # 7) Open
    webbrowser.open(f"file:///{output_path}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
