import os
import json
import requests
import networkx as nx

JAEGER_URL = "http://localhost:16686"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "log")

def fetch_traces(service="app-a"):
    """Fetch traces from Jaeger."""
    try:
        res = requests.get(f"{JAEGER_URL}/api/traces", params={"service": service, "limit": 50}, timeout=10)
        res.raise_for_status()
        return res.json().get("data", [])
    except Exception as e:
        print(f"Jaeger Error: {e}")
        return []

def build_graph(traces):
    """Build global dependency graph."""
    G = nx.DiGraph()
    for trace in traces:
        span_to_svc = {s["spanID"]: trace["processes"][s["processID"]]["serviceName"] for s in trace["spans"]}
        for s in trace["spans"]:
            child = span_to_svc[s["spanID"]]
            for ref in s.get("references", []):
                if ref["refType"] == "CHILD_OF":
                    parent = span_to_svc.get(ref["spanID"])
                    if parent and parent != child: G.add_edge(parent, child)
    return G

def save_subgraph(graph, service):
    """Save local subgraph for a service."""
    if service not in graph: return
    nodes = {service} | set(graph.predecessors(service)) | set(graph.successors(service))
    sub = graph.subgraph(nodes)
    data = {"target": service, "nodes": list(sub.nodes()), "edges": list(sub.edges())}
    path = os.path.join(LOG_DIR, f"subgraph_{service}.json")
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    print(f"[OK] Subgraph: {path}")

def main():
    traces = fetch_traces()
    if not traces: return
    G = build_graph(traces)
    for svc in G.nodes():
        if svc != "jaeger-query": save_subgraph(G, svc)

if __name__ == "__main__":
    main()
