import os
import json
from langchain_core.messages import HumanMessage
from llm_config import llm
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def deep_dive_node(state):
    """Analyze diagnostic data for a pod."""
    pod = state["current_pod"]
    log_dir = os.path.join(os.path.dirname(__file__), "..", "log")
    io_base64 = encode_image("../log/metric_profiling/consolidated_io_proxy.png")
    cpu_base64 = encode_image("../log/metric_profiling/consolidated_cpu_proxy.png")

    
    # Load Logs
    log_path = os.path.join(log_dir, f"abstracted_logs_{pod}.json")
    logs = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
            
    # Load Subgraph
    subgraph_path = os.path.join(log_dir, f"subgraph_{pod}.json")
    edges = []
    if os.path.exists(subgraph_path):
        with open(subgraph_path, "r") as f:
            edges = json.load(f).get("edges", [])
    
    predecessors = [e[0] for e in edges if e[1] == pod]
    successors = [e[1] for e in edges if e[0] == pod]

    prompt = f"""
    Analyze pod: {pod}
    Cpu_Proxy: {cpu_base64}
    IO_proxy: {io_base64}
    Logs: {json.dumps(logs[:8])}
    Callers: {predecessors}
    Callees: {successors}

    Provide a concise summary of findings and a causal hypothesis.
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print(f"Agent Deep Dive Response: {response}")
    except Exception as e:
        response = f"LLM error: {e}"
    
    # Returning partial updates for Annotated state
    return {
        "summaries": {pod: response},
        "seen_candidates": [pod]
    }