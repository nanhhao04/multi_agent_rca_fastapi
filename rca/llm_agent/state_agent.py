import operator
import json
import os
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END

from agent_rerank import rerank_node
from agent_deep_dive import deep_dive_node
from agent_remediation import remediation_node

class AgentState(TypedDict):
    initial_ranking_metrics: List[str]
    initial_ranking_traces: List[str]
    current_ranking: List[str]
    seen_candidates: Annotated[List[str], operator.add]
    iteration_count: int
    current_pod: str
    diagnostic_bundles: Dict[str, Any]
    summaries: Annotated[Dict[str, str], operator.ior]
    action: str
    next_candidate: str
    final_report: str

def set_pod_node(state: AgentState):
    """Set next pod and increment iteration."""
    return {
        "current_pod": state["next_candidate"],
        "iteration_count": state["iteration_count"] + 1
    }

def route_after_rerank(state: AgentState):
    """Route to deep dive or remediation."""
    if state["action"] == "Finish" or state["iteration_count"] >= 3:
        return "remediation"
    return "set_pod"

def build_graph():
    """Construct the LangGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("rerank", rerank_node)
    graph.add_node("set_pod", set_pod_node)
    graph.add_node("deep_dive", deep_dive_node)
    graph.add_node("remediation", remediation_node)
    graph.set_entry_point("rerank")
    graph.add_conditional_edges("rerank", route_after_rerank, {"set_pod": "set_pod", "remediation": "remediation"})
    graph.add_edge("set_pod", "deep_dive")
    graph.add_edge("deep_dive", "rerank")
    graph.add_edge("remediation", END)
    return graph.compile()

def load_initial_state(metric_path: str, trace_path: str) -> Dict[str, Any]:
    """Load initial state from ranking files."""
    initial_metrics = []
    if os.path.exists(metric_path):
        with open(metric_path, 'r') as f:
            data = json.load(f)
            initial_metrics = [item["service"] for item in data.get("ranking", [])]
    
    initial_traces = []
    if os.path.exists(trace_path):
        with open(trace_path, 'r') as f:
            data = json.load(f)
            initial_traces = [item["service"] for item in data.get("ranking", [])]
            
    # Combine rankings for initial exploration, preserving order and uniqueness
    seen = set()
    current_ranking = []
    for s in initial_metrics + initial_traces:
        if s not in seen:
            current_ranking.append(s)
            seen.add(s)
            
    return {
        "initial_ranking_metrics": initial_metrics,
        "initial_ranking_traces": initial_traces,
        "current_ranking": current_ranking,
        "seen_candidates": [],
        "iteration_count": 0,
        "current_pod": "",
        "diagnostic_bundles": {},
        "summaries": {},
        "action": "",
        "next_candidate": "",
        "final_report": ""
    }

if __name__ == "__main__":
    app = build_graph()
    
    # Paths to phase 1 and phase 2 outputs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metric_ranking_file = os.path.join(base_dir, "log", "metric_rca_ranking.json")
    trace_ranking_file = os.path.join(base_dir, "log", "trace_rca_ranking.json")

    init_state = load_initial_state(metric_ranking_file, trace_ranking_file)
    
    print(f"Loaded initial ranking metrics: {init_state['initial_ranking_metrics']}")
    print(f"Loaded initial ranking traces: {init_state['initial_ranking_traces']}")
    print(f"Merged exploration ranking: {init_state['current_ranking']}")

    result = app.invoke(init_state)
    print("\n--- FINAL REPORT ---")
    print(result.get("final_report"))