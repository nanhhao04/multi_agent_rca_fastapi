import operator
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

if __name__ == "__main__":
    app = build_graph()
    init_state = {
        "initial_ranking_metrics": ["app-a"],
        "initial_ranking_traces": [],
        "current_ranking": ["app-a"],
        "seen_candidates": [],
        "iteration_count": 0,
        "current_pod": "",
        "diagnostic_bundles": {},
        "summaries": {},
        "action": "",
        "next_candidate": "",
        "final_report": ""
    }
    result = app.invoke(init_state)
    print(result.get("final_report"))